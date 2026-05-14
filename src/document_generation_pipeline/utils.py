import asyncio
import json
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path

import requests
import yaml
from safetytooling.apis import InferenceAPI
from safetytooling.apis.batch_api import BatchInferenceAPI
from safetytooling.data_models import Prompt
from tqdm.asyncio import tqdm as atqdm

logging.getLogger("aiodns").setLevel(logging.ERROR)


def parse_tags(text: str, tag_name: str) -> str:
    """
    Parse text between opening and closing tags with the given tag name.

    Args:
        text: The text to parse
        tag_name: Name of the tags to look for (without < >)

    Returns:
        The text between the opening and closing tags, or empty string if not found
    """
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def parse_list(text: str, prefix: str = "-") -> list[str]:
    list_of_objs = text.split("\n")
    return [obj.strip().lstrip(prefix).strip() for obj in list_of_objs if obj.strip()]


def load_txt(prompt_path: str):
    with open(prompt_path) as file:
        prompt = file.read()
    return prompt


def load_json(json_path: str | Path) -> dict:
    with open(json_path) as file:
        json_data = json.load(file)
    return json_data


def load_jsonl(jsonl_path: str | Path) -> list[dict]:
    with open(jsonl_path) as file:
        jsonl_data = [json.loads(line) for line in file]
    return jsonl_data


def load_universe_contexts(path: str | Path) -> list[dict]:
    """Load universe contexts from YAML or JSONL file."""
    path = str(path)
    if path.endswith(".yaml") or path.endswith(".yml"):
        with open(path) as f:
            data = yaml.safe_load(f)
        if data is None:
            return []
        return [data] if isinstance(data, dict) else data
    return load_jsonl(path)


def save_json(json_path: str | Path, data: dict, make_dir: bool = True, **kwargs):
    if make_dir:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as file:
        json.dump(data, file, **kwargs)


def save_jsonl(jsonl_path: str | Path, data: list[dict], make_dir: bool = True):
    if make_dir:
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "w") as file:
        for item in data:
            file.write(json.dumps(item) + "\n")


def send_push_notification(title, body):
    headers = {
        "Access-Token": os.environ["PUSHBULLET_API_KEY"],
        "Content-Type": "application/json",
    }
    data = {"type": "note", "title": title, "body": body}
    response = requests.post("https://api.pushbullet.com/v2/pushes", headers=headers, json=data)
    if response.status_code != 200:
        print(f"Failed to send push notification. Status code: {response.status_code}")
        print(f"Response: {response.text}")


def wrap_in_push(fn, job_name, push_on):
    if os.environ.get("PUSHBULLET_API_KEY") is None:
        print("No PUSHBULLET_API_KEY found, skipping push notifications")

    if not push_on or os.environ.get("PUSHBULLET_API_KEY") is None:
        fn()
    else:
        try:
            fn()
            send_push_notification(f"{job_name} finished", "Job completed successfully.")
        except Exception as e:
            send_push_notification(f"{job_name} error", str(e))
            raise e


async def batch_generate(
    api: InferenceAPI = None,
    batch_api: BatchInferenceAPI = None,
    use_batch_api: bool = False,
    prompts: list[Prompt] | Prompt = None,
    model_id: str = None,
    use_tqdm: bool = True,
    n: int = 1,
    use_cache: bool | None = None,
    chunk_size: int | None = None,
    batch_id_callback: Callable[[str], None] | None = None,
    tqdm_kwargs: dict = {},
    seeds: list[int] | None = None,
    **kwargs,
) -> list[str]:

    if prompts is None:
        raise ValueError("prompts is required")
    if model_id is None:
        raise ValueError("model_id is required")

    if isinstance(prompts, Prompt):
        prompts = [prompts]

    if n > 1:
        if len(prompts) > 1:
            raise ValueError("n > 1 is not supported when > 1 prompts are provided")
        prompts = prompts * n

    if use_batch_api:

        async def batch_call(prompts: list[Prompt]):
            responses, batch_id = await batch_api(
                prompts=prompts, model_id=model_id, use_cache=use_cache or False, **kwargs
            )
            if batch_id_callback is not None:
                batch_id_callback(batch_id)
            print(f"Length of responses: {len(responses)}, batch_id: {batch_id}")
            return responses

        if chunk_size is None:
            chunk_size = len(prompts)
        raw_responses = await asyncio.gather(
            *[batch_call(prompts[i : i + chunk_size]) for i in range(0, len(prompts), chunk_size)],
        )
        responses = [item for response_list in raw_responses for item in response_list]

    # send to the api in safety tooling. caching is impliit in it
    else:
        responses = await atqdm.gather(
            *[
                api(
                    prompt=p,
                    model_id=model_id,
                    use_cache=use_cache or True,
                    **kwargs,
                    **({"seed": seeds[i]} if seeds else {}),
                )
                for i, p in enumerate(prompts)
            ],
            disable=not use_tqdm,
            **tqdm_kwargs,
        )
        responses = [r[0] for r in responses]

    return responses

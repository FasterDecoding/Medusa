import typer
import json
from transformers import Conversation
from typing_extensions import Annotated
import httpx
import tqdm
import asyncio

app = typer.Typer()


client = httpx.AsyncClient(timeout=None)

async def run(conv: Conversation, url: str):
    payload = {"model":"tgi", "messages": conv.messages}
    response = await client.post(url, json=payload)
    content = response.json()
    message = content["choices"][0]["message"]
    message.pop("name", None)
    conv.add_message(message)




def fix_source(source):
    if source and source[0]["from"] == "gpt":
        # Skip if GPT is first to talk
        source = source[1:]
    new_source = []
    for item in source:
        role = "assistant" if item["from"] == "gpt" else "user"
        content = item["value"]
        new_source.append({"role": role, "content": content})
    return new_source


async def recreate_conversation(conversation, sem, url):
    async with sem:
        conv = Conversation()
        try:
            for message in conversation[::2]:
                assert message["role"] == "user"
                conv.add_message(message)
                await run(conv, url)
        except Exception as e:
            print(e)
            pass
        return conv.messages

@app.command()
def main(
    *,
    input_filename: Annotated[str, typer.Option("--input-filename")],
    output_filename: Annotated[str, typer.Option("--output-filename")],
    url: Annotated[str, typer.Option("--url")] = "http://localhost:8080/v1/chat/completions",
    concurrency: Annotated[int, typer.Option("--concurrency")] = 64
):
    sem = asyncio.Semaphore(concurrency)
    async def _main():
        with open(input_filename, "r") as f:
            input_data = json.loads(f.read())
        conversations = [fix_source(source["conversations"]) for source in input_data]

        futures = []
        for conversation in conversations:
            future = recreate_conversation(conversation, sem, url)
            futures.append(future)

        recreated_conversations = await tqdm.asyncio.tqdm.gather(*futures)

        with open(output_filename, "w") as f:
            json.dump(recreated_conversations, f, indent=4)
    asyncio.run(_main())


if __name__ == "__main__":
    app()

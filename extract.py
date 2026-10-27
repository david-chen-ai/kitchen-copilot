"""Day 1: send one invoice/receipt image to a vision model and print what comes back.

Usage:  uv run python extract.py path/to/receipt.jpg
Needs a .env file containing:  DEEPSEEK_API_KEY=your-key-here
"""
import base64
import mimetypes
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

import json
from shape import Invoice
from pydantic import ValidationError
from pathlib import Path

schema = json.dumps(Invoice.model_json_schema(), indent=2)


load_dotenv()

# DeepSeek's API is OpenAI-compatible, so we use the openai package with a different base URL.
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)
MODEL = "deepseek-flash"  # vision-capable model per DeepSeek's vision docs

PROMPT = f"""Extract the invoice in this image.
Return ONLY a JSON object matching this JSON schema. No prose, no markdown fences.
If a value is not visible, do not guess.

{schema}"""


def call_model(image_path: str) -> str:
    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",  # images are only allowed in user messages
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }
        ],
    )
    return response.choices[0].message.content


def parse_invoice(json_str:str) -> Invoice:
    return Invoice.model_validate_json(json_str)

def extract_invoice(image_path:str) -> Invoice:
    raw = call_model(image_path)
    try:
        return parse_invoice(raw)
    except ValidationError as e:
        retry_raw = call_model(image_path, extra_message=[
            {"role": "assistant", "content": raw},
            {"role": "user", "content": f"That failed validation:\n{e}\nReturn corrected JSON only."},
        ])
        return parse_invoice(retry_raw)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python extract.py path/to/image.jpg")

    image_path = Path(sys.argv[1])
    invoice = extract_invoice(str(image_path))
    print(invoice)

    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{image_path.stem}.json"
    out_file.write_text(invoice.model_dump_json(indent=2), encoding="utf-8")
    print(f"Saved {out_file}")
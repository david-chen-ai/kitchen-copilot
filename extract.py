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

def make_client() -> OpenAI:
    load_dotenv()
    return OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )


MODEL = "deepseek-flash"  # vision-capable model per DeepSeek's vision docs

schema = json.dumps(Invoice.model_json_schema(), indent=2)

PROMPT = f"""Extract the invoice in this image.
Return ONLY a JSON object matching this JSON schema. No prose, no markdown fences.
If a value is not visible, do not guess.

{schema}"""


def call_model(image_path: str, client: OpenAI, extra_messages=None) -> str:
    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }
    ]
    if extra_messages:
            messages.extend(extra_messages)

    response = client.chat.completions.create(model=MODEL, messages=messages)
    
    return response.choices[0].message.content

def parse_invoice(json_str:str) -> Invoice:
    return Invoice.model_validate_json(json_str)

def extract_invoice(image_path:str, client: OpenAI) -> Invoice:
    raw = call_model(image_path, client)
    try:
        return check_totals(parse_invoice(raw))
    except ValidationError as e:
        retry_raw = call_model(
        image_path,
        client,
        extra_messages=[
            {"role": "assistant", "content": raw},
            {"role": "user", "content": f"That failed validation:\n{e}\nReturn corrected JSON only."},
    ],
)
        return check_totals(parse_invoice(retry_raw))

''''Compare sum of invoices to total to verify they are the same'''
def check_totals(invoice: Invoice, tolerance: float = 0.01) -> Invoice:
    items_sum = sum(item.total for item in invoice.line_items)
    if abs(items_sum - invoice.total) > tolerance:
        invoice.needs_review = True
    return invoice

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python extract.py path/to/image.jpg")

    image_path = Path(sys.argv[1])
    invoice = extract_invoice(str(image_path), make_client())
    print(invoice)

    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{image_path.stem}.json"
    out_file.write_text(invoice.model_dump_json(indent=2), encoding="utf-8")
    print(f"Saved {out_file}")
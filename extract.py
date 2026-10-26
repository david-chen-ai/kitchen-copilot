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

load_dotenv()

# DeepSeek's API is OpenAI-compatible, so we use the openai package with a different base URL.
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)
MODEL = "deepseek-flash"  # vision-capable model per DeepSeek's vision docs

PROMPT = (
    "Extract this invoice or receipt as JSON with these keys: "
    "vendor, date (YYYY-MM-DD), line_items (list of objects with "
    "description, quantity, unit_price, total), and total. "
    "Return only the JSON, no other text."
)


def extract(image_path: str) -> str:
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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python extract.py path/to/image.jpg")
    print(extract(sys.argv[1]))
import json
import pytest
from unittest.mock import Mock
from extract import extract_invoice
from pydantic import ValidationError

VALID = json.dumps({"vendor": "Acme", "date": "2026-10-01", "total": 10.0, "line_items": []})

# Items that add up to 10.00
MATCHING_ITEMS = [
    {"description": "Tomatoes", "quantity": 2, "unit_price": 3.00, "total": 6.00},
    {"description": "Olive oil", "quantity": 1, "unit_price": 4.00, "total": 4.00},
]

# Items that add up to 8.00 (will not match a total of 10.00)
MISMATCHED_ITEMS = [
    {"description": "Tomatoes", "quantity": 2, "unit_price": 3.00, "total": 6.00},
    {"description": "Olive oil", "quantity": 1, "unit_price": 2.00, "total": 2.00},
]


def make_response(text):
    """Fake the OpenAI-style response: response.choices[0].message.content"""
    message = Mock()
    message.content = text
    choice = Mock()
    choice.message = message
    response = Mock()
    response.choices = [choice]   # a real list, so [0] works
    return response

@pytest.fixture
def image(tmp_path):
    p = tmp_path / "receipt.jpg"
    p.write_bytes(b"fake-image-bytes")
    return str(p)

def test_valid_invoice_passes(image):
    fake_client = Mock()
    fake_client.chat.completions.create.return_value = make_response(invoice_json(MATCHING_ITEMS, 10.00))

    result = extract_invoice(image, fake_client)

    assert result.vendor == "Acme"
    assert fake_client.chat.completions.create.call_count == 1

def test_invalid_json_retries_once(image):
    fake_client = Mock()
    fake_client.chat.completions.create.side_effect = [
        make_response("not valid json"),          # 1st call: bad
        make_response(invoice_json(MATCHING_ITEMS, 10.00)), # 2nd call: good
    ]

    invoice = extract_invoice(image, fake_client)

    assert invoice.vendor == "Acme"
    assert fake_client.chat.completions.create.call_count == 2

def invoice_json(items, total):
    return json.dumps({
        "vendor": "Acme",
        "date": "2026-10-01",
        "total": total,
        "line_items": items,
    })

def test_totals_mismatch(image):
    fake_client = Mock()
    fake_client.chat.completions.create.return_value = make_response(
    invoice_json(MISMATCHED_ITEMS, 10.00)
    )
    invoice = extract_invoice(image, fake_client)

    assert invoice.needs_review is True

def test_totals_match(image):
    fake_client = Mock()
    fake_client.chat.completions.create.return_value = make_response(
    invoice_json(MATCHING_ITEMS, 10.00)
    )
    invoice = extract_invoice(image, fake_client)

    assert invoice.needs_review is False

def test_fails_twice(image):
    fake_client = Mock()
    fake_client.chat.completions.create.side_effect = [
        make_response("not valid json"),          # 1st call: bad
        make_response("still not valid json"),            # 2nd call: also bad
    ]

    with pytest.raises(ValidationError):
        extract_invoice(image, fake_client)

    assert fake_client.chat.completions.create.call_count == 2

def test_retry_feedback(image):
    fake_client = Mock()
    fake_client.chat.completions.create.side_effect = [
        make_response("not valid json"),
        make_response(VALID),  
    ]
    extract_invoice(image, fake_client)

    retry_messages = fake_client.chat.completions.create.call_args_list[1].kwargs["messages"]

    assert retry_messages[-2] == {"role": "assistant", "content": "not valid json"}
    assert "failed validation" in retry_messages[-1]["content"]
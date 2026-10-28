from pydantic import BaseModel

class LineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    total: float

class Invoice(BaseModel):
    vendor: str
    date: str
    line_items: list[LineItem]
    total: float
    needs_review: bool = False

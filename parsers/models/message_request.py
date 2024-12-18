from pydantic import BaseModel

class MessageRequest(BaseModel):
    operation: str 
    entity: str
    data: dict
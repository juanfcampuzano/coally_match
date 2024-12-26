from pydantic import BaseModel, Field
from typing import Optional

class CreateResumeRequest(BaseModel):
    id: str
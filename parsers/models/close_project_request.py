from pydantic import BaseModel

class CloseProjectRequest(BaseModel):
    id: str
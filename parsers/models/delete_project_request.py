from pydantic import BaseModel

class DeleteProjectRequest(BaseModel):
    id: str 
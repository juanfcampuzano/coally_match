from pydantic import BaseModel

class UpdateProjectRequest(BaseModel):
    id: str
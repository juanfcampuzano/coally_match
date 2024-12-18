from pydantic import BaseModel

class UpdateProjectRequest(BaseModel):
    id_project: str
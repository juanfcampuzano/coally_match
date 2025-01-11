from pydantic import BaseModel

class ApplyRequest(BaseModel):
    id_resume: str
    id_project: str
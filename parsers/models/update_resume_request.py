from pydantic import BaseModel

class UpdateResumeRequest(BaseModel):
    resume_id: str
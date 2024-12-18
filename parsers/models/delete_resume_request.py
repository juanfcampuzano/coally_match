from pydantic import BaseModel

class DeleteResumeRequest(BaseModel):
    resume_id: str
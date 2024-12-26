from pydantic import BaseModel

class DeleteResumeRequest(BaseModel):
    id: str
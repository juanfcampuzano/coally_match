from pydantic import BaseModel
from typing import List

class ProjectModel(BaseModel):
    experience: int
    education_level: str
    technical_skills: List[str]
    keywords: List[str]
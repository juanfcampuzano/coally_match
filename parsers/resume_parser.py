from .base_parser import BaseParser
from .models.resume_model import ResumeModel

class ResumeParser(BaseParser):
    def __init__(self):
        output_model = ResumeModel
        template="""
        Given the following resume, you will extract and classify information:

        - Main skills of the person: {main_skills}
        - Abstract of the resume: {resume_abstract}
        - Experience of the person: {experience}
        - Current position of the person: {current_position}
        - Education of the person: {education}

        You will extract the following information:

        - Time of experience of the resume (in months): if not provided, estimate it. For example, an internship means 0, senior level means 60 or 84 months.
        - Max education level reached by the person: You can only use the ones provided here: [high_school, associate, bachelor, master, doctorate] it might not be explicit in the text, but for a contact center it is typically "high_school", for an engineer or simmilar (doctor, lawyer, etc.) it is typically "bachelor" unless it specifies it requires a master or doctorate.
        - Technical skills of the person: Tools, methodologies, technologies or software that the person knows. Use the full name of the skill, using a standard name and without abbreviations. Typically these skills are proper names.
        - Keywords: List of the most relevant keywords of the resume.
        """
        super().__init__(output_model=output_model, template=template)
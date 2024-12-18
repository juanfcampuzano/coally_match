from .base_parser import BaseParser
from .models.project_model import ProjectModel

class ProjectParser(BaseParser):
    def __init__(self):
        output_model = ProjectModel
        template="""
        Given the following job offer, you will extract and classify information:


        - Name of the job offer: {project_name}
        - Description of the job offer: {project_description}
        - Majors identified by the user: {majors}
        - Hard skills identified by the user: {hard_skills}
        - Contract type: {hard_skills}

        You will extract the following information:

        - Minimum time of experience required for the job offer (in months): if not provided, estimate it. For example, an internship means 0, senior level means 60 or 84 months.
        - Education level required for the job offer: You can only use the ones provided here: [high_school, associate, bachelor, master, doctorate] it might not be explicit in the text, but for a contact center it is typically "high_school", for an engineer or simmilar (doctor, lawyer, etc.) it is typically "bachelor" unless it specifies it requires a master or doctorate. contract type "Aprendizaje SENA" suggests its an "associate". "high_school" is when you don't need to study anything to be a good candidate, for example call center agents, cashiers, stitchers, etc.
        - Technical skills required for the job: Tools, methodologies, technologies or software that the candidate must know. Use the full name of the skill, using a standard name and without abbreviations. Typically these skills are proper names.
        - Keywords: List of the most relevant keywords for the role.
        """
        super().__init__(output_model=output_model, template=template)
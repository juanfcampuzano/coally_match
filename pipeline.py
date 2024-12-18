from preprocessors import Preprocessor

class Pipeline:
    def __init__(self):
        self.pp = Preprocessor()

    def run_pipeline(self, parsed_resume, parsed_project):

        result = []
        functions = [
            self.pp.calculate_keywords_similarity,
            self.pp.calculate_skills_similarity,
            self.pp.experience_diff,
            self.pp.keywords_number,
            self.pp.education_level_score
        ]

        for function in functions:
            dict_result = function(parsed_resume=parsed_resume, parsed_project=parsed_project)
            result.append(list(dict_result.values())[0])

        scaled_result = self.pp.scale([result])
        
        return scaled_result
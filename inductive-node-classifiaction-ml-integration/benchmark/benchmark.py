from py_models.pipeline import TrainingPipeline
from py_models.model import BaseModel


class BenchmarkResult:
    def __init__(self, training_pipeline: TrainingPipeline) -> None:
        self.qualityChecks = training_pipeline.run_quality_checks()
        self.bestStats = training_pipeline.get_best_model_stats()
        self.bestModel: BaseModel = training_pipeline.get_best_model()

    def __str__(self):
        return str({'qualityChecks': str(self.qualityChecks),
                    'bestModelStats': str(self.bestStats),
                    'bestModel': str(self.bestModel)})

    def __repr__(self):
        return self.__str__()

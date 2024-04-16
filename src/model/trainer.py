import mlrun
import os
import logging
from src.utils.helpers import load_project


class ModelTrainer:
    def __init__(self, project_name, dataset_path):
        logging.info('Preprocessing component....')
        self.project_name = project_name
        self.dataset_path = dataset_path

    def runner(self):
        load_project(self.project_name)
        auto_trainer = mlrun.import_function("hub://auto_trainer")

        model_class = "sklearn.ensemble.RandomForestClassifier"
        additional_parameters = {
            "CLASS_max_depth": 2,
        }

        train_run = auto_trainer.run(
            inputs={"dataset": f"{self.dataset_path}"},
            params={
                "model_class": model_class,
                "train_test_split_size": 0.2,
                "random_state": 42,
                "label_columns": "Hazardous",
                "model_name": 'asteroid',
                **additional_parameters
            },
            handler='train',
            local=True
        )

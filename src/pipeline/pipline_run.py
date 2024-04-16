from src.preprocess.data_prep import DataPreprocessor
import logging
from src.utils.constant import (asteroid_dataset, columns_to_drop, transformed_dataset_path)
from src.utils.constant import file_path, project_name, columns_to_encode, dataset_path
from src.preprocess.run_component import DataVisualization
from src.model.trainer import ModelTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)


class MlrunPipeline:
    def __init__(self):
        """
        Initializes the MLflow Pipeline with DataPreprocessor and NeuralNetwork objects.
        """
        self.prep_data = DataPreprocessor(asteroid_dataset, columns_to_drop, transformed_dataset_path,
                                          columns_to_encode)
        self.describe = DataVisualization(file_path, project_name, columns_to_encode, dataset_path)

        self.trainer = ModelTrainer(project_name, dataset_path)

    def runner_for_mlrun_and_pipeline(self):
        try:
            logging.info("Starting the pipeline.")

            # Step 1: Data Preprocessing
            logging.info("Beginning data preprocessing.")
            self.prep_data.preprocess_data()
            logging.info("Data preprocessing completed.")
            self.describe.runner()
            logging.info("model training start")
            self.trainer.runner()
            logging.info("model training complete")




        except Exception as e:
            logging.error(f"An error occurred during pipeline execution: {str(e)}")
            raise




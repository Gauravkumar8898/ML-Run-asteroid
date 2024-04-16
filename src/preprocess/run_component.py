import mlrun
from src.utils.helpers import load_project
import os
import logging
from mlrun.errors import MLRunHTTPStatusError as error1


class DataVisualization:
    def __init__(self, file_path, project_name, columns_to_encode, dataset_path):
        self.file_path = file_path
        self.project_name = project_name
        self.columns_to_encode = columns_to_encode
        self.dataset_path = dataset_path
        logging.info('Preprocessing component....')

    def runner(self):
        """
               Executes the data visualization process.

               Raises:
                   error1: If there's an MLRun HTTP status error.
                   Exception: If any other unexpected error occurs.
        """
        try:
            project = load_project(self.project_name)
            data_gen_fn = project.set_function(func=f"{self.file_path}", name="Asteroid-dataset", kind="job",
                                               image="mlrun/mlrun",
                                               handler="data_generator")
            project.save()
            gen_data_run = project.run_function("Asteroid-dataset", local=True)

            describe_func = mlrun.import_function("hub://describe")

            describe_run = describe_func.run(
                name="Asteroid-dataset-describe",
                handler='analyze',
                inputs={"table": f"{self.dataset_path}"},
                params={"name": "Asteroid dataset", "label_column": f"{self.columns_to_encode}"},
                local=True
            )

        except ConnectionRefusedError as c1:
            raise error1

        except Exception as e:
            raise e



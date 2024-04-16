import pandas as pd
from src.utils.constant import transformed_dataset_path, dataset_path
import mlrun
import os


@mlrun.handler()
def data_generator(context):
    """a function which generates the dataset"""
    dataset = pd.read_csv(transformed_dataset_path)
    try:
        os.mkdir('artifacts')
    except:
        pass

    dataset.to_parquet(dataset_path)
    context.logger.info("saving flipkart dataframe")

import logging
import pandas as pd
from imblearn.over_sampling import SMOTE
from src.utils.helpers import (drop_columns, column_encode, load_dataset)


class DataPreprocessor:
    def __init__(self, train_data, columns_to_drop, transformed_dataset_path, columns_to_encode):
        self.file_path = train_data
        self.columns_to_drop = columns_to_drop
        self.transformed_dataset_path = transformed_dataset_path
        self.columns_to_encode = columns_to_encode

    def preprocess_data(self):
        try:
            logging.info("Loading dataset...")
            df = load_dataset(self.file_path)

            logging.info("encoding categorical variables...")
            df = column_encode(df, self.columns_to_encode)

            logging.info("Dropping Columns not to include in training")

            df = drop_columns(df, self.columns_to_drop)
            X = df.drop(columns=['Hazardous'])
            y = df['Hazardous']

            # Apply SMOTE
            smote = SMOTE()
            X_resampled, y_resampled = smote.fit_resample(X, y)

            # Combine resampled features and target into a DataFrame
            resampled_df = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled, columns=['Hazardous'])],
                                     axis=1)
            print(df.value_counts())
            resampled_df.to_csv(self.transformed_dataset_path, index=False)

        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            raise e


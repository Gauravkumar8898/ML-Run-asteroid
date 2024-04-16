from imblearn.over_sampling import SMOTE
import pandas as pd
from src.utils.constant import transformed_dataset_path

data = pd.read_csv("/home/nashtech/PycharmProjects/MLRun-asteroid/src/data/preprocessed_dataset.csv")
# Convert data to DataFrame
df = pd.DataFrame(data)

# Separate features and target
X = df.drop(columns=['Hazardous'])
y = df['Hazardous']

# Apply SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Combine resampled features and target into a DataFrame
resampled_df = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled, columns=['Hazardous'])], axis=1)
resampled_df.to_csv('/home/nashtech/PycharmProjects/MLRun-asteroid/src/data/preprocessed_dataset1.csv', index = False)
# Count the number of True and False after oversampling
print(resampled_df['Hazardous'].value_counts())

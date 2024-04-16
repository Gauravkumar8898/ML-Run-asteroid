from pathlib import Path
import pandas as pd

path = Path(__file__).parents[1]
path1 = Path(__file__).parents[2]

model_local_path = path1 / "asteroid.pkl"

# Paths
data = path / 'data'
asteroid_dataset = data / 'nasa_dataset.csv'
transformed_dataset_path = data / 'preprocessed_dataset1.csv'
dataset_path = data / "random_dataset.parquet"

plt_drop = ['Neo Reference ID', 'Name', 'Orbit ID', 'Close Approach Date',
            'Epoch Date Close Approach', 'Orbit Determination Date', 'Orbiting Body', 'Equinox']

columns_to_drop = ['Neo Reference ID', 'Name', 'Orbit ID', 'Close Approach Date',
                   'Orbit Determination Date', 'Orbiting Body', 'Equinox',
                   'Epoch Date Close Approach', 'Orbit Determination Date', 'Est Dia in KM(max)', 'Est Dia in M(min)',
                   'Est Dia in M(max)', 'Est Dia in Miles(min)'
    , 'Est Dia in Miles(max)', 'Est Dia in Feet(min)', 'Est Dia in Feet(max)',
                   'Relative Velocity km per hr', 'Miles per hour', 'Miss Dist.(lunar)',
                   'Miss Dist.(kilometers)', 'Miss Dist.(miles)']

columns_to_encode = ['Hazardous']

project_name = "asteroid"

file_path = path / "preprocess/data.py"

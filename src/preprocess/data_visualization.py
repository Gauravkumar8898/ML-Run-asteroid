from src.utils.helpers import prepare_and_plot_ts_heatmap, load_dataset, plot_correlation_matrix
from src.utils.constant import asteroid_dataset


class DataVisualization:

    def __init__(self):
        self.asteroid_data_path = asteroid_dataset

    def visualize_data(self):
        df = load_dataset(self.asteroid_data_path)
        prepare_and_plot_ts_heatmap(df)
        plot_correlation_matrix(df)

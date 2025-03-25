import pandas as pd
import numpy as np

class Results:
    def __init__(self, col_names):
        self.col_names = col_names
        self.df = pd.DataFrame()

    def update_result(self, data: np.ndarray):
        assert data.shape[-1] == len(self.col_names),\
            f"The shape of the input array doesn't match the number of columns in the results dataframe."

        # self.df = pd.DataFrame(columns=self.col_names)
        self.df = self.df._append(pd.DataFrame(data=data, columns=self.col_names), ignore_index=True)

    def save(self, filename):
        self.df.to_csv(filename, index=False)

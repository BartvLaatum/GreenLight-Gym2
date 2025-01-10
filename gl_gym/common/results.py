import pandas as pd
import numpy as np

class Results:
    def __init__(self, col_names):
        self.col_names = col_names
        self.df = pd.DataFrame()

    def update_result(self, data):
        print(data.shape)
        print(self.col_names)
        print(len(self.col_names))
        assert data.shape[-1] + 1 == len(self.col_names),\
            f"The shape of the input array doesn't match the number of columns in the results dataframe."
        
        self.df= pd.DataFrame(columns=self.col_names)

        for episode in range(data.shape[0]):
            # add the episode number to the data
            episode_data = np.concatenate((data[episode], np.full(shape=(data.shape[1], 1), fill_value=episode)), axis=1)
            self.df = self.df._append(pd.DataFrame(data=episode_data, columns=self.col_names), ignore_index=True)

    def save(self, filename):
        self.df.to_csv(filename, index=False)

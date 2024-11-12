import pandas as pd
import os

import pre_process_data._exception_handling
import numpy as np
    

class DataPreprocessor:
    def __init__(self, config):
        self.path = config.get("flow_file_paths", "new.csv")
        self.df = pd.read_csv(self.path)
        self.path_out = config.get("path_out", "new.csv")
        
    def process_data(self):
        num_regions = self.df.shape[1] 
        for i in range(2, num_regions):
            flow = self.df.iloc[:, i].values
            processed_flow = pre_process_data._exception_handling.oulier_detection(flow)
            self.df.iloc[:, i] = processed_flow

    def save_processed_data(self):

        self.df.to_csv(self.path_out, index=None)




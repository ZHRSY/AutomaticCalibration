import pandas as pd 
import numpy as np
import yaml

import pandas as pd 
import numpy as np


def load_config(config_file_path):
    with open(config_file_path, 'r') as file:
        return yaml.safe_load(file)

class Pararead:
    def __init__(self, config, model):
        self.mode = config.get('mode')
        self.model = model
        self.model_paramsnum()
        precision = config.get('precision')
        self.enlarge = 1/precision




    def variable_ul(self):
        if self.mode == 'A':
            ups = np.ones(self.nums, dtype=int) * int(self.enlarge)
            lows = np.zeros(self.nums, dtype=int) * int(self.enlarge)
        elif self.mode == 'B':
            ups = np.concatenate([np.ones(self.nums, dtype=int) * int(self.enlarge), [int(0.5 * self.enlarge), int(10 * self.enlarge)]])
            lows = np.concatenate([np.zeros(self.nums, dtype=int) * int(self.enlarge), [int(0 * self.enlarge), int(1 * self.enlarge)]])
        else:
            print('mode type error')
        # print(ups, lows,self.enlarge)
        return ups, lows
    
    def model_paramsnum(self):
        model_nums = {'xaj': 15, 'shb': 11, 'dbe': 15, 'vmrm': 18}
        self.nums = model_nums.get(self.model, 15)
        return self.nums

if __name__ == '__main__':
    config = load_config('config.yml')
    para = Pararead(config)
    ups, lows = para.variable_ul()
    print(ups,lows)
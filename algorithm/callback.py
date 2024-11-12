from pymoo.core.callback import Callback
import numpy as np
import pandas as pd


class PrintParametersCallback(Callback):
    def __init__(self):
        super().__init__()
        self.iterations = []
        self.best_r2_scores = []

    def __call__(self, algorithm):
        X = algorithm.pop.get("X")
        X_str = np.array2string(X, separator=', ')
        X = pd.DataFrame(X)
        file_path = 'parameters_callback_runoff.csv'
        X.to_csv(file_path, mode='a', header=False, index=False)
        print("本次迭代参数:")
        print(X_str)

        iteration = algorithm.n_gen
        best_r2 = -algorithm.pop.get("F").min()
        self.iterations.append(iteration)
        self.best_r2_scores.append(best_r2)
        df = pd.DataFrame({
            'Iteration': self.iterations,
            'BestR2': self.best_r2_scores
        })
        df.to_csv('optimization_results_runoff.csv', mode='a', index=False)
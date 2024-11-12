
from pymoo.core.repair import Repair  #
import numpy as np  

class ParaRepair(Repair):
    def _do(self, problem, pop, **kwargs):
        # 种群修复，确保 (para[0] + para[1]) * 0.7 - 1 < 0

        for i in range(len(pop)):
            x = pop[i]
            # print(x)
            para = x[9:11] 
            max_sum = 0.95 / 0.7 * problem.enlarge
            while (para[0] + para[1]) >= max_sum or para[0] == 0:
                para[0] = np.random.randint(0, problem.enlarge)
                para[1] = np.random.randint(0, problem.enlarge)
            x[9:11] = para
            pop[i] = x
        return pop

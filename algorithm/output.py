from pymoo.util.display.column import Column
from pymoo.util.display.output import Output
import numpy as np


class ProcessOutput(Output):
    def __init__(self,problem):
        super().__init__()
        self.f_min = Column("train_max", width=9)
        self.f_mean = Column("train_mean", width=9)
        self.f_std = Column("train_std", width=9)
        self.f_mtdm =  Column("time_dif", width=7)
        self.f_mdm = Column("max_flood_dif", width=7)
        self.f_adm = Column("flood_volume_dif", width=7)


        self.columns += [self.f_min, self.f_mean, self.f_std, self.f_adm, self.f_mtdm, self.f_mdm]

    def update(self, algorithm):
        self.more_paras(algorithm.problem)
        super().update(algorithm)
        self.f_min.set(-np.min(algorithm.pop.get("F")))
        self.f_mean.set(-np.mean(algorithm.pop.get("F")))
        self.f_std.set(np.std(algorithm.pop.get("F")))
        self.f_mtdm.set(self.max_time_diff_mean)
        self.f_mdm.set(self.max_diff_mean)
        self.f_adm.set(self.area_diff_mean)



    def more_paras(self,problem):
        self.max_time_diff_mean = problem.max_time_mean
        self.max_diff_mean = problem.max_dif_mean
        self.area_diff_mean = problem.area_diff_mean


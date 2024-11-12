from pre_process_data.preprocess import DataPreprocessor  #数据预处理
from flood_seg.flood_seg import FloodSegmentation  #场次划分
from flood_seg.classfication import RunoffClassifier  #场次聚类
import numpy as np
import yaml
import os,sys
from multiprocessing.pool import ThreadPool
import multiprocessing
from pymoo.core.problem import StarmapParallelization
from problem.problem import AutomaticCalibrationParam
from algorithm.sampling import HySampling
from algorithm.output import ProcessOutput
from pymoo.core.evaluator import Evaluator
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from algorithm.mutat import NewPM as Npm
from pymoo.operators.crossover.sbx import SBX
from pymoo.core.callback import Callback
from algorithm.repair import ParaRepair
import io
import sys
from joblib import Parallel, delayed
from problem.parall import JoblibParallelization

class DualOutput:
    def __init__(self, *outputs):
        self.outputs = outputs

    def write(self, message):
        for output in self.outputs:
            output.write(message)

    def flush(self):
        for output in self.outputs:
            output.flush()


class AutomaticCalibration:
    def __init__(self, config_file_path='./config.yml'):
        self.config_file_path = config_file_path
        self.config = self.load_config()
        self.column_num = self.config.get('column_num')
        self.mskg_pre_cl = self.config.get('mskg_columns')
        self.mode = self.config.get('mode')
        self.core = self.config.get('core')
        self.n_gen = self .config.get('n_gen')
        self.pop_size = self.config.get('pop_size')



    def target_area_cal(self,target_area):
        self.target_area = target_area

    def load_config(self):
        with open(self.config_file_path, 'r') as file:
            return yaml.safe_load(file)

    def preprocess_data(self):
        data_preprocessor = DataPreprocessor(self.config)
        data_preprocessor.process_data()
        data_preprocessor.save_processed_data()

    def segment_flood(self):
        flood_segmentation = FloodSegmentation(self.config, self.target_area)
        return flood_segmentation.process()

    def classify_flood(self, flood_result):
        classifier = RunoffClassifier(self.config, self.target_area)
        clustered_result, clustered_features = classifier.classify(flood_result)
        classifier.char(features=clustered_features)
        return clustered_result, clustered_features
    
    def runner(self):
        pool = Parallel(n_jobs=self.core, prefer="processes")
        runner = JoblibParallelization(aJoblibParallel=pool, aJoblibDelayed=delayed)

        return runner


    def run_optimization_algorithm(self, problem):

        algorithm = GA(pop_size=self.pop_size, sampling=problem.sampling, eliminate_duplicates=True, 
                    #    parallelization = runner,
                       mutation=Npm(prob=1, vtype=int), repair=ParaRepair(),
                       crossover=SBX(prob=1))
        
        res = minimize(problem, algorithm, output=ProcessOutput(problem=problem), 
                    #    elementwise_runner=runner,
                       termination=('n_gen', self.n_gen), verbose=True, callback=Callback(), save_history=True)

        return res

    def optimization(self, xaj_area, mskg_pre_cl, clustered):
        # n_proccess = self.core
        # pool = multiprocessing.Pool(n_proccess)
        # runner = StarmapParallelization(pool.starmap)
        problem = AutomaticCalibrationParam(self.config, xaj_area, mskg_pre_cl, clustered,
                                            #  elementwise_runner=self.runner()
                                             )
        model = problem.model
        enlarge = problem.enlarge
        candidates = problem.candidates()
    

        ini_candidates = self.config.get('ini_candidate')

        if ini_candidates == 'Y':
            sampling = HySampling(candidates=candidates, model=model)
            print('初始化采样')
        else:
            sampling = HySampling(candidates=None, model=model)
            print('不初始化采样')

        problem.sampling = sampling._do(problem, self.pop_size, enlarge)
        
        res = self.run_optimization_algorithm(problem)

        # pool.close()
        
        max_time_diff = problem.max_time
        max_diff = problem.max_dif
        area_diff = problem.area_diff

        problem.model_run(res.X)
        problem.para_update(res.X)
        self.problem = problem

        for entry in res.history:
            print(f"Generation: {entry.n_iter}")
            print(f"Population Size: {entry.pop_size}")
            
            for ind in entry.opt:
                print(f"本代最佳参数: {ind._X/enlarge}")
                print(f"适应度R2: {-ind._F[0]}")
            
            print("-" * 40 + "")

        print(f'最佳参数：{res.X/enlarge}')
        print(f'优化算法总耗时{np.round(res.exec_time, 2)}s')

        self.important_vars = problem.variable_gather()


        

    def plot(self):
        self.problem.result_plot()
    
    def plot_one(self, times, real, pre, rainfall,R2,ID):
        self.problem.plot_individual(times, real, pre,  rainfall,R2,ID)
        






if __name__ == '__main__':
    n_gen = 5
    pop_size = 10

    captured_output = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = DualOutput(original_stdout, captured_output)
    auto_calib = AutomaticCalibration()
    auto_calib.preprocess_data()
    if auto_calib.mode == 'A':
        for i in range(len(auto_calib.column_num)):
            auto_calib.target_area_cal(auto_calib.column_num[i])
            flood_result = auto_calib.segment_flood()
            clustered_result, clustered_features = auto_calib.classify_flood(flood_result)
            for j in range(len(clustered_result)):

                print(f'特征{i+1}参数率定：')
                clustered_one = clustered_result[j]
                auto_calib.optimization(xaj_area= auto_calib.column_num[i], mskg_pre_cl=auto_calib.mskg_pre_cl[j], clustered=clustered_one)

    elif auto_calib.mode == 'B':
            for i in range(len(auto_calib.mskg_pre_cl)):
                auto_calib.target_area_cal(auto_calib.mskg_pre_cl[i])
                flood_result = auto_calib.segment_flood()
                clustered_result, clustered_features = auto_calib.classify_flood(flood_result)

                for j in range(len(clustered_result)):
                    print(f'特征{i+1}参数率定：')
                    clustered_one = clustered_result[j]
                    auto_calib.optimization(xaj_area= auto_calib.column_num[i],mskg_pre_cl=auto_calib.mskg_pre_cl[i], clustered=clustered_one)

    sys.stdout = original_stdout
    with open('report.log', 'w') as report_file:
        report_file.write(captured_output.getvalue())


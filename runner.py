from Autocali import  DualOutput
import sys,io
import pandas as pd


class CalibrationRunner:
    def __init__(self, auto_calib):
        self.auto_calib = auto_calib
        self.captured_output = io.StringIO()
        self.original_stdout = sys.stdout

        sys.stdout = DualOutput(self.original_stdout, self.captured_output)

    def run(self):
        self.auto_calib.preprocess_data()

        if self.auto_calib.mode == 'A':
            self.run_mode_a()
        elif self.auto_calib.mode == 'B':
            self.run_mode_b()
        sys.stdout = self.original_stdout
        self.save_report()
        

    def run_mode_a(self):
        self.history = []
        for i in range(len(self.auto_calib.column_num)):
            self.auto_calib.target_area_cal(self.auto_calib.column_num[i])
            flood_result = self.auto_calib.segment_flood()
            clustered_result, clustered_features = self.auto_calib.classify_flood(flood_result)
            for j in range(len(clustered_result)):
                print(f'特征{i+1}参数率定：')
                clustered_one = clustered_result[j]
                self.auto_calib.optimization(
                    xaj_area=self.auto_calib.column_num[i],
                    mskg_pre_cl=self.auto_calib.mskg_pre_cl[j],
                    clustered=clustered_one
                )
            self.problem = self.auto_calib.problem
            self.history.append(self.auto_calib.important_vars)
            self.plot_history(i)

    def run_mode_b(self):
        self.history = []
        for i in range(len(self.auto_calib.mskg_pre_cl)):
            self.auto_calib.target_area_cal(self.auto_calib.mskg_pre_cl[i])
            flood_result = self.auto_calib.segment_flood()
            clustered_result, clustered_features = self.auto_calib.classify_flood(flood_result)
            for j in range(len(clustered_result)):
                print(f'特征{i+1}参数率定：')
                clustered_one = clustered_result[j]
                self.auto_calib.optimization(
                    xaj_area=self.auto_calib.column_num[i],
                    mskg_pre_cl=self.auto_calib.mskg_pre_cl[i],
                    clustered=clustered_one
                )
            self.history.append(self.auto_calib.important_vars)
            self.problem = self.auto_calib.problem
            self.plot_history(i)


    def save_report(self):
        with open('report.log', 'w') as report_file:
            report_file.write(self.captured_output.getvalue())

    def plot(self):
        self.auto_calib.plot()

    def plot_one(self, times, real, pre, rainfall,R2,ID):
        self.problem.plot_individual(times, real, pre,  rainfall,R2,ID)


    def plot_history(self,i):
        _, standard = self.auto_calib.important_vars
        standard_df = pd.DataFrame(standard)
        for j, (new_pre, new_res, new_time, rain) in enumerate(zip(self.problem.pre_val, self.problem.tem_real_value, self.problem.clu_time, self.problem.clu_rainfall)):
            new_time = pd.to_datetime(new_time)
            ID = f'section_{i+1}_{j+1}'
            R2 = standard_df['确定性系数'][j]
            self.problem.plot_individual(new_time, new_pre, new_res, rain, R2,ID)


        


if __name__ == '__main__':
    auto_calib = AutomaticCalibration()
    calibration_runner = CalibrationRunner(auto_calib)
    calibration_runner.run()

from pymoo.core.problem import ElementwiseProblem
import matplotlib.pyplot as plt
import time
from matplotlib.widgets import RangeSlider
import plotly.express as px
import plotly.graph_objects as go
import io
import plotly.io as pio
from multiprocessing import Lock

from problem.para import *
import sys,os
import numpy as np
import pandas as pd 
from sklearn.metrics import r2_score
sys.path.append("..")
from hydro.python.calc import xaj



class AutomaticCalibrationParam(ElementwiseProblem):

    def __init__(self, config,column_num,mskg_pre_cl,clustered, **kwargs):
        self.config = config
        self.cas_path = config.get('cas_path')
        self.pre_path = config.get('pre_path')
        self.cas_read()
        para = Pararead(config, self.model)
        self.ups,self.lows = para.variable_ul()

        self.nums_para = para.model_paramsnum()
        n_var = len(self.ups)
        self.mode = config.get('mode')
        self.column_num = column_num
        self.mskg_pre_cl = mskg_pre_cl
        self.clustered = clustered
        precision = config.get('precision')
        self.enlarge = 1/precision


        self.flow_file_paths = config.get("path_out", "new.csv")
        self.rain_file_paths = config.get("rain_file_paths", "basin_rainfall.csv")

        df = pd.read_csv(self.flow_file_paths, encoding="utf-8-sig", encoding_errors='ignore')
        self.time = df.iloc[:, 0].values
        # print(self.time)

        self.rainfall_df = pd.read_csv(self.rain_file_paths, encoding="utf-8-sig", encoding_errors='ignore')
        self.rainfall = self.rainfall_df.iloc[:, self.column_num].values



        self.real_value = np.array(pd.read_csv(config.get('path_out')).iloc[:,self.column_num + 1])

        # n_ieq_constr =  1
        n_ieq_constr = np.shape(self.real_value)[0] +1

        
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=n_ieq_constr, xl=self.lows, xu=self.ups, vtype=int, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = -self.model_run(x)
        para = x[9:11] / self.enlarge
        g1 = (para[0] + para[1])* 0.7 - 1
        g2 = [-val for val in self.pre_const]
        out["G"] = np.hstack([g2, g1])


    def model_run(self,x):
   
        self.para_update(x)
        # print(1)

        original_stdout = sys.stdout 
        sys.stdout = open(os.devnull, 'w')  
        try:
            xaj() 
            # time.sleep(10)
             
        finally:
            sys.stdout.close() 
            sys.stdout = original_stdout  
        # xaj()

        self.data_read()
        result = self.out_dataframe()
        

        return np.mean(result)

    def out_dataframe(self):
        result = []
        area_diff = []
        max_dif = []
        max_time = []

        area_diff_all = []
        max_dif_all = []
        self.recorded_results = []
        for new_pre, new_res,new_time in zip(self.pre_val, self.tem_real_value, self.clu_time):
            result_tem = r2_score(new_res, new_pre)
            result.append(result_tem)

            
            area_di = 1- abs(np.trapz(new_pre) - np.trapz(new_res))/np.trapz(new_res)
            area_diff.append(area_di)

            max_dif_tem = 1 - abs(max(new_pre) - max(new_res)) / max(new_res)
            max_dif.append(max_dif_tem)

            max_time_tem =abs(np.where(new_pre == max(new_pre))[0][0] - np.where(new_res == max(new_res))[0][0])
            max_time.append(max_time_tem)

            # 洪量
            area_pre = np.trapz(new_pre)
            area_real = np.trapz(new_res)
            area_relative_error = abs(area_pre - area_real) / area_real

            # 洪峰
            max_val_pre = max(new_pre)
            max_val_real = max(new_res)
            local_max_time_pre = np.where(new_pre == max_val_pre)[0][0]
            local_max_time_real = np.where(new_res == max_val_real)[0][0]
            max_time_pre  = new_time[local_max_time_pre]
            max_time_real = new_time[local_max_time_real]

            max_relative_error = abs(max_val_pre - max_val_real) / max_val_real
            totel_hours = len(new_pre)




            max_time_diff = abs(local_max_time_pre - local_max_time_real)

            area_diff_all.append((area_pre, area_real, area_relative_error))
            max_dif_all.append((max_val_pre, max_val_real, max_time_pre,max_time_real, max_time_diff))
            self.recorded_results.append({
                '场次序列': len(self.recorded_results) + 1,
                '模拟洪量(106m3)': round(area_pre, 2),
                '实测洪量(106m3)': round(area_real, 2),
                '洪量相对误差': round(area_relative_error, 2),
                '模拟洪峰(m3/s)': round(max_val_pre, 2),
                '实测洪峰(m3/s)': round(max_val_real, 2),
                '洪峰相对误差': round(max_relative_error,2),
                '模拟峰现时间': max_time_pre,
                '实测峰现时间': max_time_real,
                '洪水总时间': totel_hours,
                '误差h': round(max_time_diff, 2),
                '确定性系数': round(result_tem, 3)
            })





        #面积百分比
        self.area_diff = area_diff
        self.area_diff_mean = np.mean(area_diff)
        #洪峰流量误差
        self.max_dif = max_dif
        self.max_dif_mean = np.mean(max_dif)
        #峰现时间差
        self.max_time = max_time
        self.max_time_mean = np.mean(max_time)
        return result

    def data_read(self):
        hymodel_pre = pd.read_csv(self.pre_path + f'/{self.ID}/output/Q_basin.csv').iloc[:, self.column_num + 1]
        msmodel_pre = pd.read_csv(self.pre_path + f'/{self.ID}/output/Q_MSKG.csv').iloc[:, self.mskg_pre_cl + 1]

        self.tem_real_value,self.hymodel_pre,self.msmodel_pre,self.clu_time,self.clu_rainfall = [],[],[],[],[]
        for i in range(len(self.clustered)):
                self.tem_real_value.append(self.real_value[self.clustered[i][0]:self.clustered[i][1]])
                self.hymodel_pre.append(hymodel_pre[self.clustered[i][0]:self.clustered[i][1]])
                self.msmodel_pre.append(msmodel_pre[self.clustered[i][0]:self.clustered[i][1]])
                self.clu_time.append(self.time[self.clustered[i][0]:self.clustered[i][1]])
                self.clu_rainfall.append(self.rainfall[self.clustered[i][0]:self.clustered[i][1]])

        if self.mode == 'A':
            self.pre_const = hymodel_pre
            self.pre_val = self.hymodel_pre
        else:
            self.pre_const = msmodel_pre
            self.pre_val = self.msmodel_pre



    def x_to_params(self, x):
        hydro_params = x[:self.nums_para] / self.enlarge
        mskg_params = x[self.nums_para:] / self.enlarge if self.mode != 'A' else []
        return hydro_params, mskg_params

    def para_update(self,x):
        hydro_params, mskg_params = self.x_to_params(x)


        if mskg_params == []:

            hy_para_df = pd.read_csv(self.para_path + '/Param_Hydro_flood.csv')
            hy_para_df.iloc[self.column_num - 1, -self.nums_para - 3:-3] = hydro_params
            hy_para_df.to_csv(self.para_path + '/Param_Hydro_flood.csv', index=False)
        else:
            hy_para_df = pd.read_csv(self.para_path + '/Param_Hydro_flood.csv')
            mskg_para_df = pd.read_csv(self.para_path + '/Param_MSKG_flood.csv')

            hy_para_df.iloc[self.column_num - 1, -self.nums_para - 3: - 3] = hydro_params
            mskg_para_df.iloc[self.mskg_pre_cl - 1, 3:5] = mskg_params
            hy_para_df.to_csv(self.para_path + '/Param_Hydro_flood.csv', index=False)
            mskg_para_df.to_csv(self.para_path + '/Param_MSKG_flood.csv', index=False)

    def cas_read(self):
        with open(self.cas_path, 'r') as fr:
            lines = [line.strip() for line in fr if line.strip() and not line.startswith("/")]

        for line in lines:
            try:
                key, content = map(str.strip, line.split("=", 1))
                if key == '计算编号':
                    self.ID = content

                elif key == '模型参数编号':
                    # self.para_path = './hydro/params/' + content
                    self.para_path = self.config.get('para_path')

                elif key == '水文模型':
                    self.model = content



                else:
                    pass
            except ValueError:
                print('率定cas文件:', line)

    

    def candidates(self):
            if self.model == 'xaj':
                hy_para_df = pd.read_csv(self.para_path + '/Param_Hydro_flood.csv')
                hydro_values = hy_para_df.iloc[self.column_num - 1, -self.nums_para-3:-3].values
                scaled_values = hydro_values * self.enlarge
                self.candidates = np.array([scaled_values], dtype=int)

         
                if self.mode == 'B':

                    mskg_para_df = pd.read_csv(self.para_path + '/Param_MSKG_flood.csv')
                    mskg_params = mskg_para_df.iloc[self.mskg_pre_cl - 1, 3:5].values
                    scaled_values_ms = mskg_params * self.enlarge
 
                    self.candidates = np.concatenate([scaled_values, scaled_values_ms]).reshape(1, -1)
                    # print(self.candidates)            


            return self.candidates




    def result_plot(self):
        """

        :param self.time: 时间
        :param self.real_value: 真实流量
        :param self.rainfall: 降雨量
        :param self.clustered: 处理后洪水间隔
        """
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=self.time, y=self.real_value, mode='lines', name='流量 (m3/s)', line=dict(color='green', width=2)))
        fig.add_trace(
            go.Scatter(x=self.time, y=self.pre_const , mode='lines', name='pre_Flow (m3/s)', line=dict(color='orange', width=2)))

        fig.add_trace(go.Bar(x=self.time, y=self.rainfall, name='降雨量 (mm)',
                     marker=dict(color='blue', line=dict(width=2, color='blue')), yaxis='y2'))
        yaxis = 'y2'

        for interval in self.clustered:
            if interval[1] < len(self.time):
                fig.add_vrect(x0=self.time[interval[0]], x1=self.time[interval[1]], fillcolor='pink', opacity=0.3, line_width=0)
        fig.update_layout(
            title=f'result',
            title_x=0.5,
            xaxis_title='时间',
            yaxis_title='流量 (m3/s)',
            title_font=dict(size=40),
            xaxis=dict(
            title_font=dict(size=25),
            tickfont=dict(size=15)
            ),
            yaxis=dict(
            title_font=dict(size=25),
            tickfont=dict(size=15),
            range=[0, max(self.real_value) * 2]
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            # plot_bgcolor='white',
            # paper_bgcolor='white'
        )
        fig.update_layout(
            yaxis2=dict(
            title='降雨量 (mm)',
            overlaying='y',
            side='right',

            # autorange='reversed',
            range=[max(self.rainfall) * 2,0],
            title_font=dict(size=25),
            tickfont=dict(size=15)
            )
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        fig.show()



    def variable_gather(self):
        vars = {
            '模型': self.model,
            'ID': self.ID,
            '参数精度': self.enlarge,
            '预测断面': self.column_num
        }

        return vars, self.recorded_results
    

    def plot_individual(self,times, real, pre,  rainfall,R2,ID):
        plt.rcParams.update({'font.size': 20})
        rainfall = np.array(rainfall)
        plt.rcParams['font.family'] = 'SimHei'  

        fig, ax1 = plt.subplots(figsize=(24, 12))
        ax1.plot(times, real, color='blue', label='实测流量(m3/s)', linewidth=2)
        ax1.set_ylim([0, max(real) * 2])
        plt.text(0.95, 0.95, f'R2: {R2:.3f}', fontsize=30, ha='right', va='top', transform=ax1.transAxes)
        ax1.plot(times, pre, color='red', label='预测流量(m3/s)', linewidth=2)
        ax1.set_ylabel('流量(m3/s)', fontsize=40)

        ax2 = ax1.twinx()
        ax2.bar(times, rainfall, color='olive', label='降雨量(mm)', width=0.03)
        ax2.set_ylabel('降雨量(mm)', fontsize=40)
        ax2.yaxis.set_inverted(True)
        ax2.set_ylim(max(rainfall) * 3, 0)
        ax1.xaxis_date()
        ax1.set_xlabel('时间', fontsize=40)
        plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
        ax1.legend(loc='upper left', bbox_to_anchor=(0, 0.95))
        plt.title(f'率定结果', fontsize=40, fontweight='bold')
        plt.savefig(f'./result/{ID}.png', dpi=100, bbox_inches='tight')
        return fig


    def plot_individual_1(self, times, real, pre, rainfall, R2):
        pio.kaleido.scope.default_format = "jpeg" 
        rainfall = np.array(rainfall)

        fig = go.Figure()

        # Add real flow trace
        fig.add_trace(go.Scatter(x=times, y=real, mode='lines', name='实测流量(m3/s)', line=dict(color='blue', width=2)))

        # Add predicted flow trace
        fig.add_trace(go.Scatter(x=times, y=pre, mode='lines', name='预测流量(m3/s)', line=dict(color='red', width=2)))

        # Add rainfall bar trace
        fig.add_trace(go.Bar(x=times, y=rainfall, name='降雨量(mm)', marker=dict(color='olive'), width=0.03))

        # Update layout
        fig.update_layout(
            title=f'率定结果 (R2: {R2:.3f})',
            xaxis_title='时间',
            yaxis_title='流量(m3/s)',

            yaxis=dict(range=[0, max(real) * 2]),
            # yaxis2=dict(overlaying='y', side='right', range=[max(rainfall) * 3, 0]),
            legend=dict(x=0, y=1, traceorder='normal'),
            font=dict(size=20)
        )
        fig.update_layout(
            yaxis2=dict(
            title='降雨量 (mm)',
            overlaying='y',
            side='right',

            # autorange='reversed',
            range=[max(self.rainfall) * 2,0],
            title_font=dict(size=25),
            tickfont=dict(size=15)
            )
        )

        image_io = io.BytesIO()
        fig.write_image(image_io, format="jpeg")

        return image_io
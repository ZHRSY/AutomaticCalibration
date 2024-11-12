import os,sys
import time
import multiprocessing as mp
from copy import deepcopy

sys.path.append(f"{os.getcwd()}/python/runoff_forecast")
from runoff_forecast.HydroCase import HydroCase,Parallel_Calc
from runoff_forecast.print_colors import *

from reservoir import *


def Calc(HCalc,Floods,caseid):
    HC:HydroCase = deepcopy(HCalc)

    HC.calcid = caseid
    HC.Initialize_path()

    HC.time = Floods[caseid][0]
    HC.settings['预见结束日期'] = Floods[caseid][1]
    HC._read_warnup()
    HC._read_forecast_period()
    HC.Get_calc_period()
    if HC.settings['水库调洪'] == 'YES':
        HC.Read_Reservoir_definition()

    #读取输入的降雨json文件
    HC.Read_raindata_by_json()
    #生成输入给水文模型的子流域降雨量和蒸发量文件
    HC.Generate_PRCP_TS()
    HC.Generate_EVAP_TS()

    HC.Calc(debug=False) 
    print(f'{caseid} ok')
    return 'ok'

if __name__ == '__main__':

    HCalc = HydroCase("settings_calc.cas")
    HCalc.Resv_routing = {
        "XQ":XQ_Routing,
        "BTW":BTW_Routing,
    }

    Floods ={
        "01":["2024-09-27 00:00:00","2024-09-28 00:00:00"],
        "02":["2024-09-27 00:00:00","2024-09-29 00:00:00"],
        "03":["2024-09-27 00:00:00","2024-09-30 00:00:00"],
        "04":["2024-09-27 05:00:00","2024-09-30 00:00:00"],
    }
    
    args_list = []
    for caseid in Floods:
        args_list.append((HCalc,Floods,caseid))
    Parallel_Calc(Calc,args_list,4)
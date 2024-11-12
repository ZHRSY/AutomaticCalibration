import os,sys
import time

current_dir = os.getcwd()
os.chdir('./hydro')
sys.path.append(f"{os.getcwd()}/python")

from runoff_forecast.HydroCase import HydroCase
from runoff_forecast.print_colors import *
from reservoir import *
os.chdir(current_dir)

def xaj():
    os.chdir('./hydro')
    sys.path.append(f"{os.getcwd()}/python")
    print(f"\n{BLUE}PYTHONPATH:{RESET} {sys.path}\n")

    Routings_func = {
        "XQ":XQ_Routing,
        "BTW":BTW_Routing,
    }

    s = time.time()
    HCalc = HydroCase("settings_calc.cas",log=True)
    HCalc.Resv_routing = Routings_func

    #读取输入的降雨json文件
    # HCalc.Read_raindata_by_json()
    # #生成输入给水文模型的子流域降雨量和蒸发量文件
    # HCalc.Generate_PRCP_TS()
    # HCalc.Generate_EVAP_TS()

    HCalc.Calc(savePRCP="./input_raindata.csv",debug=False) 

    e = time.time()
    print("%.2f s"%(e-s))
    os.chdir(current_dir)
    



if __name__ == '__main__':
    xaj()
    os.chdir(current_dir)
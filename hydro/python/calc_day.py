import os,sys
import time

sys.path.append(f"{os.getcwd()}/python/runoff_forecast")
from runoff_forecast.HydroCase import HydroCase
from runoff_forecast.print_colors import *

from reservoir import *


print(f"\n{BLUE}PYTHONPATH:{RESET} {sys.path}\n")

Routings_func = {
    "XQ":XQ_Routing,
    "BTW":BTW_Routing,
}

# os.chdir(r"D:\python_files\来水预报标准化")
# os.chdir("/media/fla/disk/fla/来水预报标准化")

s = time.time()
HCalc = HydroCase("settings_calc_day.cas",log=True)
HCalc.Resv_routing = Routings_func

#读取输入的降雨json文件
HCalc.Read_raindata_by_json()
#生成输入给水文模型的子流域降雨量和蒸发量文件
HCalc.Generate_PRCP_TS()
HCalc.Generate_EVAP_TS()

HCalc.Calc(savePRCP="./input_raindata_day.csv",debug=False) 

e = time.time()
print("%.2f s"%(e-s))
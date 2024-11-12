import os,sys
import time

sys.path.append(f"{os.getcwd()}/python/runoff_forecast")
from runoff_forecast.HydroCase import HydroCase
from runoff_forecast.print_colors import *

from reservoir import *

print(f"\n{BLUE}PYTHONPATH:{RESET} {sys.path}\n")

s = time.time()
HCalc = HydroCase("settings_calc.cas",log=True)

#读取输入的降雨json文件
HCalc.Read_raindata_by_json()
#生成输入给水文模型的子流域降雨量
HCalc.Generate_PRCP_TS("./arearain")

e = time.time()
print("%.2f s"%(e-s))
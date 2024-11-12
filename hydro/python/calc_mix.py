import os,sys
import time

sys.path.append(f"{os.getcwd()}/python/runoff_forecast")
from runoff_forecast.HydroCase import HydroCase
from reservoir import *

print(f"\n{BLUE}PYTHONPATH:{RESET} {sys.path}\n")

Routings_func = {
    "XQ":XQ_Routing,
    "BTW":BTW_Routing,
}

s = time.time()
HCalc = HydroCase("settings_calc_mix.cas",log=True)
HCalc.Resv_routing = Routings_func

#生成历史降雨文件
HCalc.Read_raindata_by_json()
HCalc.Generate_PRCP_TS()
#生成未来降雨文件
HCalc.input = "./input/mesh.json"
HCalc.input_type = "mesh" #临时更改降雨输入类型
HCalc.Read_raindata_by_json()
HCalc.Generate_PRCP_TS(HCalc.pathdict['补充数据'])

#生成蒸发文件
HCalc.Generate_EVAP_TS()

HCalc.input_type = "station" #更改回原来的降雨输入类型
HCalc.Calc(savePRCP="./input_raindata_mix.csv") 

e = time.time()
print("%.2f s"%(e-s))
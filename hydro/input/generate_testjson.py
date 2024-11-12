#!/usr/bin/python3
# @ 作者: Flavien, 2023/9/12
# @ 用途: 将开发给到的输入转化为水文模型的输入（网格化降雨~>子流域降雨|站点降雨~>子流域降雨）

from mesh_rain import *
import os,time,argparse
from datetime import datetime
from scipy.interpolate import interp1d


station_info = np.loadtxt("./shp/雨量站位置.txt",delimiter=",",skiprows=1,dtype=str)
station_points = np.array(station_info[:,1:3],dtype=float)
station_names = station_info[:,0]

num = len(station_points)
rains = np.zeros((num,60))
for i in range(num):
    rains[i,25:25+24] = [
                0 , 0, 0, 3, 4, 5, 8, 16, 20, 13,
                12, 7, 5, 3, 0, 0, 0, 0,  0,  0,
                0 , 0, 0, 0]

RData_station = Generate_example_jsonfile(rains,station_points,station_names,
                                         "./input/stations.json")
print(RData_station.Tseries)
print(RData_station.rel_Tseries)
RData_station.Extrac_raindata(4,"./input/rain_station4.csv")

gridshp = "./shp/网格降雨/网格V2_4545.shp"
points = Get_Grid_points(gridshp)
num_points = len(points)
rains = np.zeros((num_points,60))

for i in range(380,385):
    rains[i,25:25+24] = [
                0 , 0, 0, 3, 4, 5, 8, 16, 20, 13,
                12, 7, 5, 3, 0, 0, 0, 0,  0,  0,
                0 , 0, 0, 0]
    
for i in range(393,397):
    rains[i,25:25+24] = [
                0 , 0, 0, 3, 4, 5, 8, 16, 20, 13,
                12, 7, 5, 3, 0, 0, 0, 0,  0,  0,
                0 , 0, 0, 0]
for i in range(368,372):
    rains[i,25:25+24] = [
                0 , 0, 0, 3, 4, 5, 8, 16, 20, 13,
                12, 7, 5, 3, 0, 0, 0, 0,  0,  0,
                0 , 0, 0, 0]
for i in range(343,347):
    rains[i,25:25+24] = [
                0 , 0, 0, 3, 4, 5, 8, 16, 20, 13,
                12, 7, 5, 3, 0, 0, 0, 0,  0,  0,
                0 , 0, 0, 0]
for i in range(318,322):
    rains[i,25:25+24] = [
                0 , 0, 0, 3, 4, 5, 8, 16, 20, 13,
                12, 7, 5, 3, 0, 0, 0, 0,  0,  0,
                0 , 0, 0, 0]
RData = RainData(rains,points,RData_station.Tseries)
RData.points = np.round(Transform_CRS(RData.points,4545,4326),6)
RData.Generate_jsonfile("./input/mesh.json")
RData.Extrac_raindata(4,"./input/rain_mesh4.csv")

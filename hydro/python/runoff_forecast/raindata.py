
import os,json
import numpy as np
from datetime import datetime,timedelta
from typing import Tuple
from shapely.geometry import MultiPoint,Point
# from pykrige import OrdinaryKriging
# import matplotlib.pyplot as plt

class RainData():
    """点位降雨数据对象

    Args:
        rains (np.ndarray): 降雨数据，二维数组（点位x时间）
        points (np.ndarray): 点位数据，二维数组（点位x2）
        Tseries (Tuple[list,np.ndarray,None], optional): 时间序列，默认为None

    Raises:
        ValueError: 时间序列的长度与降雨数据的时间长度不一致
    """
    def __init__(self,rains:np.ndarray,points:np.ndarray,
                 Tseries:Tuple[list,np.ndarray,None]=None):

        self.isinit = True #是否示例变量初始化

        self.rains       = rains #所有点位的降雨过程
        self.avgrain     = None  #平均降雨

        self.points          = points   #点位坐标
        self.num_points      = len(points) #点位个数
        self.points_geo      = MultiPoint(self.points) #点位几何对象
        self.points_name     = None   #降雨点位名称
        self.points_dict     = dict() #降雨点位字典（以点位坐标元组为键值，返回对应点位的雨量过程）
        self.points_namedict = None   #降雨点位字典（以点位名称为键值，返回对应点位的雨量过程）
        self.EPSG            = None #点的坐标系
        for i,point in enumerate(points):
            self.points_dict[ tuple(point) ] = rains[i]
        
        self.time_unit   = 3600 #单位秒
        self.tform       = "%Y-%m-%d %H:%M:%S" #年月日时分秒格式
        self.num_time    = rains.shape[1] #时刻数
        self.rel_Tseries = None #时间序列，相对时间，单位小时
        self.Tseries     = Tseries  #时间序列，年月日时分秒

        #处理时间序列
        if isinstance(Tseries,(list,np.ndarray)): #如果给定时间序列
            if len(Tseries) != rains.shape[1]: 
                raise ValueError(
                    f"Wrong Tseries: len {len(Tseries)}, expected {rains.shape[1]}."
                )
            if isinstance(Tseries, list):
                Tseries = np.array(Tseries,dtype=str)
                self.Tseries = Tseries
        else: #如果没有给定时间序列，则赋值为同长度的空字符数组
            self.Tseries = np.array([""]*self.num_time )

        self.isinit = False #非类变量初始化
        
    def __setattr__(self, key, value):
        # 属性注册
        self.__dict__[key] = value
        if (key == "Tseries"):
            if (value is not None) and (value[0] != ""):
                ststamp = datetime.strptime(value[0], self.tform).timestamp() #起始时间
                rel_Tseries = []
                for t in value:
                    tstamp = datetime.strptime(t, self.tform).timestamp() - ststamp
                    rel_Tseries.append(tstamp)
                rel_Tseries = np.array(rel_Tseries, dtype=int)/self.time_unit
                self.__dict__['rel_Tseries'] = rel_Tseries

        if not self.isinit: 
            if key == 'points': #更改属性points，将联动更改属性points_geo和points_dict
                points = value
                self.__dict__['num_points'] = len(points)
                self.__dict__['points_geo'] = MultiPoint(points)
                new_points_dict = dict()
                for i,point in enumerate(points):
                    new_points_dict[ tuple(point) ] = self.rains[i]
                self.__dict__['points_dict'] = new_points_dict
            elif key == 'rains': #更改属性rains，将联动更改属性num_time,points_dict和points_namedict
                self.__dict__['num_time'] = np.size(value,1)
                
                new_points_dict = dict()
                for i,point in enumerate(self.points):
                    new_points_dict[ tuple(point) ] = value[i]
                self.__dict__['points_dict'] = new_points_dict

                if self.points_name is not None:
                    new_namedict = dict()
                    for i,name in enumerate(self.points_name):
                        new_namedict[ name ] = value[i]
                    self.__dict__['points_namedict'] = new_namedict

            elif key == 'points_name':
                namedict = dict()
                for i,name in enumerate(value):
                    namedict[ name ] = self.rains[i]
                self.__dict__['points_namedict'] = namedict

    def Calc_avgrains(self,rains:np.ndarray, weight:np.ndarray):
        #rains:  各雨量站/子流域雨量过程数组 2-D array(点位维度x时间维度)
        #weight: 各雨量站/子流域权重 1-D array | 2-D array(点位维度x1)
        self.avgrain = np.dot(rains.T, weight)/np.sum(weight)
        return self.avgrain

    def Get_data_by_points(self,points):
        data = np.zeros((len(points),self.num_time))
        for i,point in enumerate(points):
            if str(type(point)) != "<class 'tuple'>":
                point = tuple(point)
            data[i] = self.points_dict[point]
        return data

    def Get_data_by_names(self,names):
        data = np.zeros((len(names),self.num_time))
        for i,name in enumerate(names):
            data[i] = self.points_namedict[name]
        return data

    def Extract_raindata(self,index:int, output:Tuple[str,None]=None):
        # print("extracting raindata at time frame %d ..."%index)

        RD_points  = self.points
        RD_rains   = self.rains
        num_time   = self.num_time
        if index in range(num_time):
            data_at_T = np.column_stack((RD_points,RD_rains[:,index]))
            if output is not None:
                np.savetxt(output,data_at_T,delimiter=',',fmt='%.6f',encoding="utf-8")
        else:
            data_at_T = None
    
        return data_at_T


    #将雨量数据输出成json文件
    def Generate_jsonfile(self, output):
        #获取数据
        Tseries = self.Tseries
        rains   = self.rains
        num_points = self.num_points
        points  = self.points
        lon     = points[:,0]
        lat     = points[:,1]

        #要写入JSON文件的数据
        wdata = {
            "rainMeshs": []
        }
        for i in range(num_points):
            #每个点位的坐标和降雨过程数据
            if self.points_name is not None:
                data_atapoint = {
                    "name": self.points_name[i],
                    "lon": lon[i],
                    "lat": lat[i],
                    "meshedatas": []
                }                
            else:
                data_atapoint = {
                    "lon": lon[i],
                    "lat": lat[i],
                    "meshedatas": []
                }
            for t,v in zip(Tseries,rains[i]):
                data_atapoint["meshedatas"].append(
                    {
                        "time": t,
                        "value": v
                    }
                )
            #所有点位的数据
            wdata['rainMeshs'].append(
                data_atapoint
            )

        #将数据写入到json文件
        with open(output, 'w', encoding="utf-8") as json_file:
            json.dump(wdata, json_file, ensure_ascii=False)

    def Export(self,output):
        if self.points_name is None:
            points_name = ['p%d'%i for i in range(self.num_points)]
        else:
            points_name = self.points_name

        Header = ['时间'] + list(points_name)
        data = np.column_stack(( self.Tseries, self.rains.T ))
        np.savetxt( output,data,delimiter=",",fmt='%s',encoding="utf-8",
                    header=','.join(item for item in Header),
                    comments='')
        

    def HourP2DayP(self,debug=False):
        if self.Tseries is None:
            raise ValueError(f"Tseries should have values, not None.")
        
        dtform = self.tform[:self.tform.index("%d")+2]
        collection = {}
        for i,t in enumerate(self.Tseries):
            DT = datetime.strptime(t,self.tform)
            if (DT.hour==0) and (DT.minute==0) and (DT.second==0):
                day = t
                if debug: print(t,day)
                if day not in collection:
                    collection[day] = [i]
                else:
                    collection[day].append(i)
            else:
                day = DT.strftime(dtform) #年月日字符串
                day_DT = datetime.strptime(day,dtform) + timedelta(hours=24)
                if debug: print(t,day_DT)
                day = day_DT.strftime(self.tform) #年月日时分秒字符串，但时分秒为零
                if day not in collection:
                    collection[day] = [i]
                else:
                    collection[day].append(i)
        
        day_Tseries = np.array( list(collection.keys()),dtype=str )
        day_rains = np.zeros( (self.num_points,len(day_Tseries)) )
        for i,day in enumerate(collection):
            selection = np.array(collection[day],dtype=int)
            day_rains[:,i] = np.sum( self.rains[:,selection],axis=1 )
            if debug: print(day,selection)
            
        self.time_unit = 3600*24
        self.rains = day_rains
        self.Tseries = day_Tseries
            

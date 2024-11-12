import os,re,shutil,sys
import numpy as np
from datetime import datetime, timedelta
import multiprocessing as mp
from typing import Any,Callable
sys.path.append(f"{os.getcwd()}/python/runoff_forecast/")
from mesh_rain import *
from regulation import Reservoir_Reg
from Hydro_MSKG import *
from print_colors import *


class HydroCase():
    def __init__(self,casefile:str,log:bool=False) -> None:
        self.log:bool = log

        self.settings:dict = {} #计算设置
        self.casfile:str = casefile #计算设置文件

        self.dtform:str = "" #在设置tform后会自动有值，年月日
        self.tform:str  = "%Y-%m-%d %H:%M:%S" #年月日时分秒

        self.param_dir:str   = "" #模型参数文件夹
        self.paramid:str     = "" #模型参数编号
        self.model:str       = "" #水文模型, xaj、shb、dbe或vmrm
        self.init_dir:str    = "" #初始状态文件夹
        self.output:str      = "" #计算输出文件夹
        self.calcid:str      = "" #计算编号
        self.mode:str        = "" #日计算还是次洪计算，day或flood
        self.input:str       = "" #计算输入
        self.input_type:str  = "" #计算输入类型，站点雨量还是网格雨量，station或mesh
        self.pP2aP:str       = "" #点雨量转子流域面雨量的方式（只在输入雨量为站点雨量时起作用），IDW(mesh)或voro

        self.time:str        = "" #计算时刻，年月日时分秒
        #当'预热起始日期'和'预热期'同时存在时为'duration';
        #当只存在'预热起始日期'时为'date';
        #当只存在'预热期'时为'duration';
        self.warmup_type:str = "" #预热方式
        self.warmup:int      = "" #预热期，单位时
        self.init_state:str  = "" #初始状态标记
        self.Hydro_L:int     = "" #预见期
        self.final_date:str  = "" #计算终止日期
        self.calc_period:list|None = None #计算时段，年月日
        self.calc_period_complete:list|None = None #计算时段，年月日时分秒
        self.time_array:np.ndarray|None = None #计算时间序列
        self.relT_array:np.ndarray|None = None #相对计算时间序列
        self.currentT_index:int|None = None #当前时刻在计算时间序列中的索引

        self.mesh_parafile:str  = "" #网格化降雨参数文件
        self.Hydro_parafile:str = "" #水文参数文件
        self.MSKG_parafile:str  = "" #马斯京跟参数文件

        self.RData: RainData|None = None #点位雨量数据
        self.EinMonth:dict = {} #月蒸发数据
        self.pathdict:dict = {} #文件路径字典

        self.names_subbasin:list = [] #子流域名称列表
        self.names_station:list  = [] #雨量测站名称列表
        self.names_Hydro:list    = [] #水文计算代号列表，如[H0,H1,H2,...]
        self.num_stations:int = 0 #站点个数
        self.num_subbasins:int = 0 #子流域个数
        self.Hydro_Dt:int = 0 #水文时间步长
        self.Hydro_initfile:str = "" #水文初始状态文件

        self.MSKG_def_type:str = '' #马斯京根定义方式
        self.num_MSKGs:int = 0 #马斯京根模型个数
        self.names_MSKG:list = [] #马斯京根计算代号列表，如[M0,M1,M2,...]
        self.MSKG_input:list[list] = [] #各马斯京根模型的结构/输入
        self.MSKG_input_leng:list[list] = [] #各马斯京根模型汇入口之间的距离
        self.MSKG_input_loc:list[list]  = [] #各马斯京根模型汇入口的里程
        self.MSKG_initfile:str = "" #马斯京根初始流量文件
        self.MSKG_init:np.ndarray = {} #马斯京根初始流量
        self.MSKG_para:np.ndarray|None = None #马斯京根参数
        self.MSKG_para_seg:np.ndarray|None = None #马斯京根分段参数

        self.num_Resv:int = 0 #水库个数
        self.names_Resv:list = [] #水库计算代号列表，由水库名组成，如[XQ,BTW,...]
        self.Resv_input:dict = {} #水库输入
        self.Reservoir:dict = {} #水库实例字典
        self.Resv_initfile:str = "" #水库初始状态文件
        self.Resv_init:dict = {} #水库初始状态
        self.Resv_histfile:str = "" #历史泄洪文件
        self.Resv_fill_way:str = "" #历史泄洪流量填补方式
        self.Resv_history_outflow:dict = {} #插补后的历史泄洪过程字典
        self.is_UDF_outflow:bool = False #是否用户自定义水库输入
        self.UDF_outflow:dict = {} #用户自定义水库输入
        self.Resv_routing:dict = {} #水库调度函数字典，外部提供

        self.to_calc:list = [''] #所有将计算的模型代号，默认存在代号''，即零数组

        #读取计算设置
        self.Read_settings() 


    def __setattr__(self, name:str, value:Any) -> None:
        self.__dict__[name] = value
        if name == "tform":
            self.__dict__["dtform"] = value[:value.index("%d")+2]
        

    def Read_settings(self):
        """读取设置，赋值示例属性

        In:
            self.settings (dict): 计算设置
        Out:
            ALL
            
        """
        #默认设置
        self.settings = {
            '参数文件目录默认':'YES',
            '马斯京根':'NO',
            '水库调洪':'NO',
        }
        #读取设置
        self._read_setting_file()
        self.settings['EPSG'] = int(self.settings['EPSG'])
        #赋予设置
        self.param_dir   = self.settings['模型参数文件夹']
        self.paramid     = self.settings['模型参数编号']
        self.model       = self.settings['水文模型']
        self.init_dir    = self.settings['初始状态文件夹']
        self.output      = self.settings['计算输出文件夹']
        self.calcid      = self.settings['计算编号']
        self.mode        = self.settings['计算模式']
        self.input       = self.settings['计算输入']
        self.time        = self.settings['计算时刻']
        self.input_type  = self.settings['计算输入类型']

        #确定点面雨量转换方式
        if self.input_type == 'station':
            self.pP2aP   = self.settings['点面雨量转换方式']
        elif self.input_type == 'mesh':
            pass #pP2aP默认为空字符串''
        else:
            raise ValueError(f"input_type {self.input_type} not supported!")
        
        #确定计算步长
        if self.mode == 'flood':
            self.Hydro_Dt = 1
        elif self.mode == 'day':
            self.Hydro_Dt = 24
        else:
            raise ValueError(f"mode {self.mode} not supported!")
        if self.log: print(f"{BLUE}mode:{RESET} {self.mode}")

        #读取预热期信息
        self._read_warnup()
        #读取预见期信息
        self._read_forecast_period()
        #确定计算时段
        self.Get_calc_period()

        #检查模型参数和初始状态文件夹是否存在，初始化输出文件夹
        for p in [self.param_dir,self.init_dir]:
            if not os.path.exists(p): raise FileNotFoundError(f"not such path {p}.")
        param_path = f"{self.param_dir}/{self.paramid}"
        if not os.path.exists(param_path): raise FileNotFoundError(f"not such path {param_path}.")
        self.Initialize_path()

        #读取和检查子流域和雨量测站信息
        self._read_subbasin_info()
        self._read_station_info()
        #读取月蒸发文件
        self.Read_EinMonth()

        #判断水文初始状态文件
        self.Hydro_initfile = f"{self.pathdict['初始状态']}/States_{self.init_state}.csv" #水文状态文件
        if not os.path.exists(self.Hydro_initfile): raise FileNotFoundError(self.Hydro_initfile)
        if self.log: print(f"{BLUE}use Hydro initfile{RESET} - {self.Hydro_initfile}")

        #读取马斯京根设置
        if self.settings['马斯京根'] == 'YES':
            #初始状态文件
            self.MSKG_initfile  = f"{self.pathdict['初始状态']}/MSKGQs_{self.init_state}.csv" #马斯京根状态文件
            if not os.path.exists(self.MSKG_initfile): raise FileNotFoundError(self.MSKG_initfile)
            if self.log: print(f"{BLUE}use MSKG initfile{RESET} - {self.MSKG_initfile}")
            #定义文件
            self.Read_MSKG_definition()
        elif self.settings['马斯京根'] == 'NO':
            pass
        else:
            raise ValueError(f"'马斯京根'={self.settings['马斯京根']} not supported")

        #读取水库调洪设置
        if self.settings['水库调洪'] == 'YES':
            #初始状态文件
            # self.Resv_initfile  = f"{self.pathdict['初始状态']}/Resvs_{self.init_state}.csv" #水库状态文件
            self.Resv_initfile = self.settings['水库当前状态文件']
            if not os.path.exists(self.Resv_initfile): raise FileNotFoundError(self.Resv_initfile)
            if self.log: print(f"{BLUE}use Resv initfile{RESET} - {self.Resv_initfile}")
            #定义文件
            self.Read_Reservoir_definition()
        elif self.settings['水库调洪'] == 'NO':
            pass
        else:
            raise ValueError(f"'水库调洪'={self.settings['水库调洪']} not supported")
        
        #打印将计算的代号
        if self.log: print(f"{BLUE}To calc:{RESET} {self.to_calc}")

        #检查各模型的输入
        if self.settings['马斯京根'] == 'YES':
            self.Check_inputs(self.MSKG_input) #检查输入是否存在
        if self.settings['水库调洪'] == 'YES':
            self.Check_inputs(self.Resv_input) #检查输入是否存在
        if (self.settings['马斯京根'] == 'YES') and (self.settings['水库调洪'] == 'YES'):
            self.Check_MSKG_and_Resv_inputs() #检查水库调洪和马斯京根输入是否存在“互为输入”的现象
        
        if self.log: print("="*80)

    def Initialize_path(self,check=True):
        """设置各类文件夹，检查文件夹及子文件夹是否存在，确定参数文件的实际路径

        Args:
            check (bool, optional): 是否检查文件夹的存在与否
        
        In:
            self.init_dir
            self.param_dir
            self.paramid
            self.output
            self.calcid

        Out:
            self.pathdict
            self.settings

        """

        #设置各类文件夹
        self.pathdict['初始状态'] = f"{self.init_dir}"  #初始状态文件夹
        self.pathdict['模型参数'] = f"{self.param_dir}/{self.paramid}"  #模型参数文件夹
        self.pathdict['计算输出'] = f"{self.output}"                #计算输出文件夹
        self.pathdict['计算编号'] = f"{self.output}/{self.calcid}"  #计算编号文件夹
        self.pathdict['整理结果'] = f"{self.output}/{self.calcid}/collect_data" #提取json文件的储存路径
        self.pathdict['模型输入'] = f"{self.output}/{self.calcid}/input"  #模型输入文件夹
        self.pathdict['模型输出'] = f"{self.output}/{self.calcid}/output" #模型输出文件夹
        self.pathdict['补充数据'] = f"{self.output}/{self.calcid}/addition" #降雨蒸发补充数据文件夹

        #检查文件夹的存在与否
        if check:
            for p in self.pathdict:
                if not os.path.exists(self.pathdict[p]):
                    os.makedirs(self.pathdict[p],exist_ok=True) #即使文件夹存在，也不引发OSError

        #在计算输出文件夹下记录使用的模型参数编号
        with open(f"{self.pathdict['计算编号']}/{self.paramid}.txt",'w',encoding="utf-8") as fw:
            pass

        #若参数文件目录默认为YES，则调整参数文件路径为实际路径；若否，则无需调整
        if self.settings['参数文件目录默认'] == 'YES':
            if (self.input_type=='station') and (self.pP2aP=='IDW(mesh)'):
                self.mesh_parafile  = f"{self.pathdict['模型参数']}/{self.settings['网格化降雨参数文件'][2:]}"
            self.Hydro_parafile = f"{self.pathdict['模型参数']}/{self.settings['水文参数文件'][2:]}"
            if self.settings['马斯京根'] == 'YES':
                self.MSKG_parafile = f"{self.pathdict['模型参数']}/{self.settings['马斯京根参数文件'][2:]}" 
        elif self.settings['参数文件目录默认'] == 'NO':
            pass
        else:
            raise ValueError(f"{self.settings['参数文件目录默认']} should be YES or NO.")


    def _read_setting_file(self):
        """读取设置文件

        In:
            self.casfile (str): 计算设置文件
        Out:
            self.settings (dict): 计算设置

        """
        with open(self.casfile,'r',encoding="utf-8") as fr:
            lines = fr.readlines()
        for line in lines:
            line = line.strip() #去掉首尾空白
            if (line == "") or (line[0] == "/"):
                continue
            key,content = line.split("=")
            self.settings[key.strip()] = content.strip().strip("'\"")
        if self.log: print(f"{BLUE}read setting file:{RESET} {self.casfile}")

    def _read_duration(self,string:str) -> int:
        if '天' in string:
            duration  = int(string.replace('天','')) * 24
        elif '时' in string:
            duration  = int(string.replace('时',''))
        else:
            duration  = int(string)
        return duration

    def _read_warnup(self):
        """读取预热期信息

        In:
            self.settings (dict): 计算设置
            self.mode (str): 日洪计算还是次洪计算
        Out:
            self.warmup_type (str):
            self.warmup (int): 
            self.init_state (str)
            
        """
        if ('预热起始日期' in self.settings) and ('预热期' in self.settings):
            self.warmup_type = "duration"
            self.warmup  = self._read_duration(self.settings['预热期'])
            self.init_state  = self.settings['预热起始日期']
        elif ('预热起始日期' in self.settings):
            self.warmup_type = "date"
            t_DT = datetime.strptime(self.time,self.tform)
            try:
                wu_DT = datetime.strptime(self.settings['预热起始日期'], self.dtform)  
            except Exception as e:
                print(e)
                raise ValueError(
                    "wrong type of warmup date or provide warmup duration as additional info."
                )
            self.warmup = int( (t_DT.timestamp()-wu_DT.timestamp())/3600 )
            self.init_state  = self.settings['预热起始日期']
        elif ('预热期' in self.settings):
            self.warmup_type = "duration"
            self.warmup = self._read_duration(self.settings['预热期'])
            t_DT = datetime.strptime(self.time,self.tform)
            wu_DT = t_DT - timedelta(hours=self.warmup)
            self.init_state  = wu_DT.strftime( self.dtform )
            if self.mode == 'day': #日计算时初始状态文件的日期应该提前一天
                DT = datetime.strptime(self.init_state,self.dtform) - timedelta(hours=24)
                self.init_state = DT.strftime( self.dtform )
        else:
            raise ValueError("预热起始日期 or 预热期 should be defined.")
        if self.log: print(f"{BLUE}warmup type:{RESET}{self.warmup_type} {BLUE}warmup:{RESET}{self.warmup}h")

    def _read_forecast_period(self):
        """读取预见期信息

        In:
            self.settings (dict): 计算设置
        Out:
            self.Hydro_L (int):
            self.final_date (str):
            
        """
        if ('计算终止日期' in self.settings) and ('预见期' in self.settings):
            raise ValueError("both '计算终止日期' and '预见期' are given.")
        elif ('计算终止日期' in self.settings):
            t_DT = datetime.strptime(self.time,self.tform)
            ft_DT = datetime.strptime(self.settings['计算终止日期'], self.tform)
            self.Hydro_L = int( (ft_DT.timestamp()-t_DT.timestamp())/3600 )
            self.final_date = self.settings['计算终止日期']
        elif ('预见期' in self.settings):
            t_DT = datetime.strptime(self.time,self.tform)
            self.Hydro_L = self._read_duration(self.settings['预见期'])
            ft_DT = t_DT + timedelta(hours=self.Hydro_L)
            self.final_date = ft_DT.strftime( self.tform )
        if self.log: print(f"{BLUE}forcast period:{RESET} {self.Hydro_L}h")

    def Get_calc_period(self):
        """根据预热类型和预热信息，以及水文单位时长，确定计算时段
        
        In:
            self.warmup_type (str): 预热类型
            self.warmup (int): 预热期
            self.Hydro_L (int): 水文单位时长

        Out:
            self.calc_period (list): 计算时段，年月日
            self.calc_period_complete (list): 计算时段 年月日时分秒
            self.time_array (np.ndarray): 计算时间序列
            self.relT_array (np.ndarray): 计算相对时间序列
        
        """
        self.calc_period = ['','']
        self.calc_period_complete = ['','']
        t_DT = datetime.strptime(self.time,self.tform)
        #结束时刻
        ft_DT = t_DT + timedelta(hours=self.Hydro_L)
        self.calc_period[1] = ft_DT.strftime( self.dtform )
        self.calc_period_complete[1] = ft_DT.strftime( self.tform )
        #开始时刻
        if self.warmup_type == "date":
            self.calc_period[0] = self.settings['预热起始日期']
            wu_DT = datetime.strptime(self.settings['预热起始日期'], self.dtform)
        elif self.warmup_type == "duration":
            wu_DT = t_DT - timedelta(hours=self.warmup)
            self.calc_period[0] = wu_DT.strftime( self.dtform )
            self.calc_period_complete[0] = wu_DT.strftime( self.tform )
        
        num_time = int( (ft_DT.timestamp() - wu_DT.timestamp())/self.Hydro_Dt/3600 ) + 1
        self.time_array = []
        self.relT_array = np.zeros( num_time,dtype=int )
        for i in range(num_time):
            DT = wu_DT + timedelta(hours=self.Hydro_Dt*i)
            DT_string = DT.strftime(self.tform)
            self.time_array.append(DT_string)
            self.relT_array[i] = (DT.timestamp() - t_DT.timestamp())/3600/self.Hydro_Dt
            if DT_string == self.time:
                self.currentT_index = i
        self.time_array = np.array(self.time_array,dtype=str)
        if self.currentT_index is None:
            raise ValueError(f"the current time {self.time} not found in {self.time_array}.")

        if self.log: print(f"{BLUE}calculation period:{RESET} {self.calc_period[0]} ~ {self.calc_period[1]}") #年月日
        if self.log: print(f"{BLUE}calculation period:{RESET} {self.calc_period_complete[0]} ~ {self.calc_period_complete[1]}") #年月日时分秒

    def _read_subbasin_info(self):
        """读取子流域信息，检查水文参数文件和子流域划分文件中的子流域数目是否匹配

        In:
            self.settings (str): 计算设置
        
        Out:
            self.names_subbasin (list): 子流域名称列表
            self.num_subbasins (int): 子流域个数
            self.names_Hydro (list): 水文计算代号列表，如[H0,H1,H2,...]
            self.to_calc (list): 所有将计算的模型代号，默认存在代号''，即零数组
        
        """
        #读取子流域名称
        names_subbasin_sorted,order = ReadTable(
            self.settings['子流域划分文件'],
            self.settings['子流域名称字段'],
            self.settings['子流域排序字段']
        )
        self.names_subbasin = names_subbasin_sorted
        self.num_subbasins = len(names_subbasin_sorted)

        #水文参数文件里面的子流域个数和子流域划分文件里的子流域个数
        names_subbasin_in_Param = np.loadtxt(self.Hydro_parafile,
                                            delimiter=",",skiprows=1,usecols=(0),
                                            encoding="utf-8",dtype=str)
        if len(names_subbasin_in_Param) != len(names_subbasin_sorted): 
            raise ValueError(
                f"""unmatched number of sub-basins in 
                {len(names_subbasin_sorted)} ~> {self.settings['子流域划分文件']}
                {len(names_subbasin_in_Param)} ~> {self.Hydro_parafile}"""
            )

        self.names_Hydro = ['H%d'%i for i in range(self.num_subbasins)]
        for item in self.names_Hydro: 
            self.to_calc.append(item) 

        if self.log: print(f"{BLUE}sub-basins{RESET}({self.num_subbasins}):")
        if self.log: print(f"    {self.names_subbasin}")

    def _read_station_info(self):
        """读取雨量测站信息，检查网格化降雨参数文件和泰森多边形文件中的雨量测站数目是否匹配

        In:
            self.settings (str): 计算设置
        
        Out:
            self.names_station (list): 雨量测站名称列表
            self.num_stations (int): 站点个数
        
        """
        names_station_sorted,order = ReadTable(
            self.settings['泰森多边形文件'],
            self.settings['雨量测站名称字段'],
            self.settings['雨量测站排序字段']
        )
        self.names_station = names_station_sorted
        self.num_stations = len(names_station_sorted)
        #网格化降雨参数文件里面的站点个数和泰森多边形文件里的站点个数
        if self.mesh_parafile != '':
            names_station_in_Param = np.loadtxt(self.mesh_parafile,
                                                delimiter=",",skiprows=1,usecols=(0),
                                                encoding="utf-8",dtype=str)
            if len(names_station_in_Param) != len(names_station_sorted): 
                raise ValueError(
                    f"""unmatched number of stations in 
                    {len(names_station_sorted)} ~> {self.settings['泰森多边形文件']}
                    {len(names_station_in_Param)} ~> {self.mesh_parafile}"""
                )
            for station in self.names_station:
                if station not in names_station_in_Param:
                    raise ValueError(f"{station} not in {self.mesh_parafile}")
        
            
        if self.log: print(f"{BLUE}stations{RESET}({self.num_stations}):")
        if self.log: print(f"    {self.names_station}")


    def Read_EinMonth(self):
        """读取月蒸发文件

        In:
            self.settings (dict): 计算设置

        Out:
            self.EinMonth (dict): 月蒸发数据
        """
        Evap_info = np.loadtxt(self.settings['月蒸发文件'],
                               delimiter=',',usecols=(0,1),encoding="utf-8")
        self.EinMonth =  {}
        for i in range(Evap_info.shape[0]):
            self.EinMonth[int(i)] = Evap_info[i,1]


    def Read_MSKG_definition(self):
        """读取马斯京根设置

        In:
            self.settings (dict): 计算设置

        Out:
            self.MSKG_def_type (str): 马斯京根定义方式
            self.MSKG_input (list): 各马斯京根模型的结构/输入
            self.MSKG_input_leng (list): 各马斯京根模型汇入口之间的距离
            self.MSKG_input_loc (list): 各马斯京根模型汇入口的里程
            self.names_MSKG (list): 马斯京根计算代号列表，如[M0,M1,M2,...]
            self.num_MSKGs (int): 马斯京根模型个数
            self.MSKG_para (np.ndarray): 马斯京根参数
            self.MSKG_para_seg (np.ndarray): 马斯京根分段参数
            self.MSKG_init (np.ndarray): 马斯京根初始流量
            self.to_calc (list): 所有将计算的模型代号，默认存在代号''，即零数组

        """
        self.MSKG_input = []
        self.MSKG_input_leng = []
        self.MSKG_input_loc = []

        param_value = []
        self.names_MSKG = []
        def_file = self.settings['马斯京根结构文件']
        with open(def_file,'r',encoding='utf-8') as fr:
            lines = fr.readlines()
        iter = -1
        for line in lines:
            #跳过空白行
            line = line.strip() #去掉首尾空白
            if (line == "") or ('#' in line):
                continue
            #预处理
            key,content = line.split("=")
            key = key.strip()
            #马斯京根定义方式
            content = content.strip()
            if key == 'def type':
                self.MSKG_def_type = content
                continue
            #马斯京根参数
            self.names_MSKG.append(key) #马斯京跟模型代号
            iter += 1
            param_value.append([])
            self.MSKG_input.append([])
            items = content.split(',')
            for item in items:
                item = ''.join(s for s in item.split())
                rm = re.match("(.*)\(([0-9]+)\)",item)
                self.MSKG_input[iter].append(rm.group(1))
                param_value[iter].append(rm.group(2))
                
        if self.MSKG_def_type == "distance":
            self.MSKG_input_leng = param_value
            self.MSKG_input_loc = Convert_seglength_to_x(self.MSKG_input_leng)
        elif self.MSKG_def_type == "chainage":
            self.MSKG_input_loc = param_value
        else:
            raise ValueError(f"def type {self.MSKG_def_type} is not supported.")
        

        names_MSKG_in_Param = list(np.loadtxt(self.MSKG_parafile,
                                delimiter=',',skiprows=1,usecols=(0),dtype=str))
        if len(names_MSKG_in_Param) != len(self.MSKG_input):
            raise ValueError(
                f"""unmatched number of MSKGs in 
                {len(names_MSKG_in_Param)} ~> {self.MSKG_parafile}
                {len(self.MSKG_input)} ~> {self.settings['马斯京根结构文件']}"""              
            )
        for item in self.names_MSKG:
            if item not in names_MSKG_in_Param:
                raise ValueError(
                    f"{item} not in {self.MSKG_parafile}"
                )

        self.num_MSKGs = len(self.names_MSKG)

        #读取和处理马斯京根参数
        self.MSKG_para = np.loadtxt(self.MSKG_parafile,
                            delimiter=',',skiprows=1,usecols=(1,2,3,4),ndmin=2,
                            encoding="utf-8") #马斯京根参数 DeltaT,KE,XE,n
        self.MSKG_para_seg  = Convert_MSKGpara_to_CCC(self.MSKG_para) #各马斯京根模型的单河段参数

        #读取初始状态
        self.MSKG_init = np.loadtxt(self.MSKG_initfile,skiprows=1,ndmin=1)  #马斯京根初始流量

        self.to_calc += self.names_MSKG

        # if self.log: print(self.MSKG_input)
        # if self.log: print(self.MSKG_input_loc)
        if self.log: print(f"{BLUE}read MSKG definition.{RESET}")

            
    def Read_Reservoir_definition(self):
        """读取水库设置

        In:
            self.settings (dict): 计算设置

        Out:
            self.Resv_input (dict): 水库输入
            self.Reservoir (dict): 水库实例字典
            self.names_Resv (list): 水库计算代号列表，由水库名组成，如[XQ,BTW,...]
            self.num_Resv (int): 水库个数
            self.is_UDF_outflow (str): 是否用户自定义水库输入
            self.UDF_outflow (dict): 用户自定义水库输入
            self.Resv_fill_way (str): 历史泄洪流量填补方式
            self.Resv_histfile (str): 历史泄洪文件
            self.Resv_history_outflow (dict): 插补后的历史泄洪过程字典
            self.to_calc (list): 所有将计算的模型代号，默认存在代号''，即零数组

        """
        #读取水库输入
        self.Resv_input = {}
        with open(self.settings['水库结构文件'],'r') as fr:
            lines = fr.readlines()
        for line in lines:
            line = "".join(item for item in line.split()) #去除所有空白字符
            if (line == "") or ('#' in line):
                continue
            key,content = line.split("=")
            inputs = re.match("Reg\(([a-zA-Z0-9+]+)\)",content).group(1)
            if '+' in inputs:
                self.Resv_input[key] = inputs.split('+')
            else:
                self.Resv_input[key] = [inputs]
        
        #读取水库定义文件
        self.Reservoir = {}
        for key in self.settings:
            if '水库定义文件' not in key:
                continue
            ResReg = Reservoir_Reg(self.settings[key],log=self.log)
            self.Reservoir[ResReg.NAME] = ResReg
            if self.log: print(f"{BLUE}read reservoir {ResReg.NAME} definition.{RESET}")
        self.names_Resv = list(self.Reservoir.keys())
        self.num_Resv = len(self.names_Resv)

        #读取水库初始状态
        self.Resv_init = {} 
        Resv_init_info = np.loadtxt(self.Resv_initfile,
                                    delimiter=',',skiprows=1,
                                    ndmin=2,dtype=str)
        for i in range(Resv_init_info.shape[0]):
            Rname = Resv_init_info[i,0]
            self.Resv_init[Rname] = np.array(Resv_init_info[i,1:],dtype=float) #[Qout,SL]
        for r in self.Reservoir:
            if r not in Resv_init_info[:,0]:
                raise ValueError(f"initial condition of Resv {r} should be given.")

        #是否自定义泄洪
        if self.settings['水库自定义泄洪']=='YES':
            self.is_UDF_outflow = True
        elif self.settings['水库自定义泄洪']=='NO':
            self.is_UDF_outflow = False
        else:
            raise ValueError(f"'水库自定义泄洪'={self.settings['水库自定义泄洪']} not supported")

        if self.is_UDF_outflow:
            self.UDF_outflow = {}
            outflow_info = np.loadtxt(self.settings['水库自定义泄洪文件'],
                                      delimiter=',',ndmin=2,dtype=str,
                                      encoding='utf-8')[:,1:]
            for i,r in enumerate(outflow_info[0,:]):
                if r not in self.names_Resv: raise ValueError(
                    f"Resv {r} is not defined but appears in {self.settings['水库自定义泄洪文件']}."
                )
                if len(outflow_info[1:,i]) < self.Hydro_L: raise ValueError(
                    f"insufficient given outflow of {len(outflow_info[1:,i])} v.s. Hydro_L {self.Hydro_L}"
                )
                last_index = int(self.Hydro_L/self.Hydro_Dt) + 1
                self.UDF_outflow[r] = np.array(outflow_info[1:last_index,i],dtype=float)
        
        #读取历史泄洪数据
        self.Resv_fill_way = self.settings['历史水库泄洪流量填补方式']
        if self.Resv_fill_way == 'auto':
            pass
        elif self.Resv_fill_way == 'history':
            self.Resv_histfile = self.settings['历史水库泄洪流量文件']
            self.Read_history_outflow_by_json()
        else:
            raise ValueError(f"Resv_fill_way {self.Resv_fill_way} is not supported.")

        for item in self.names_Resv: 
            self.to_calc.append(item) 

    def Check_MSKG_and_Resv_inputs(self):
        """检查马斯京根和水库输入

        In:
            self.MSKG_input (list): 各马斯京根模型的结构/输入
            self.names_Resv (list): 水库计算代号列表，由水库名组成，如[XQ,BTW,...]
            self.Resv_input (dict): 水库输入

        """
        for iter,iMSKG_input in enumerate(self.MSKG_input):
            for item in iMSKG_input:
                #子流域出口断面流量或其他马斯京根出口断面流量作为输入
                if item not in self.names_Resv: 
                    continue
                #水库出库流量作为输入
                if f"M{iter}" in self.Resv_input[item]:
                    raise ValueError(
                    f"""Resv <{item}> is an input of M{iter},
                    but M{iter} is also an input of Resv <{item}>."""
                )

    def Check_inputs(self,inputs:Tuple[list,dict]):
        """检查马斯京根/水库输入

        In:
            self.to_calc (list): 所有将计算的模型代号，默认存在代号''，即零数组

        """
        for i in inputs:
            if isinstance(i,list): #马斯京根输入
                input = i
            elif isinstance(i,str): #水库输入
                input = inputs[i]
            else:
                raise ValueError("wrong type of inputs.")
            for item in input:
                if '+' in item:
                    subitem = item.split('+')
                    for si in subitem:
                        if si not in self.to_calc: raise ValueError(f"input <{si}> is not defined.")
                else:
                    if item not in self.to_calc: raise ValueError(f"input <{item}> is not defined.")

    def Read_raindata_by_json(self,debug=False):
        """读取json格式的降雨数据

        In:
            self.input_type (list): 所有将计算的模型代号，默认存在代号''，即零数组
            self.input (str): 计算输入
            self.pathdict (dict): 文件路径字典
            self.mode (str): 日计算还是次洪计算
        
        Out:
            self.RData (RainData): 点位雨量数据

        """
        if self.input_type == "station":
            RData = Read_raindata(self.input) #读取站点降雨json文件
            filename = os.path.basename(self.input)
            filename = os.path.splitext(filename)[0]
            collect_file = f"{self.pathdict['整理结果']}/{filename}.csv"
            RData.Export(collect_file)
            if self.log: print(f'{BLUE}raindata:{RESET} {RData.Tseries[0]} - {RData.Tseries[-1]}')
            if self.log: print(f"{BLUE}create{RESET} {collect_file}")
            if len(self.names_station) != RData.num_points: #站点降雨
                if self.log: print(
                    f"""{YELLOW}unmatched number of stations in
                    {len(self.names_station)} ~> {self.settings['泰森多边形文件']}
                    {len(RData.num_points)} ~> {self.input},
                    there will be an error when using 'voro' as pP2aP method.{RESET}"""   
                )
            
            if self.mode == 'day':
                RData.HourP2DayP(debug=debug) #时降雨转化为日降雨
                collect_file = f"{self.pathdict['整理结果']}/{filename}_day.csv"
                RData.Export(collect_file)
                if self.log: print(f"{BLUE}create{RESET} {collect_file}")
        elif self.input_type == "mesh":
            RData = Read_raindata(self.input)
            if self.mode == 'day':
                RData.HourP2DayP(debug=debug) #时降雨转化为日降雨
        
        self.RData = RData


    def Read_history_outflow_by_json(self):
        """读取json格式的历史泄洪数据

        In:
            self.Resv_histfile (list): 历史泄洪文件
            self.time (str): 计算时刻，年月日时分秒 
            self.tform (dict): 年月日时分秒
            self.Reservoir (dict): 水库实例字典
            self.Hydro_Dt (int): 水文时间步长
            self.pathdict (dict): 文件路径字典
            self.warmup (int): 预热期
            self.time_array (np.ndarray): 计算时间序列
            self.relT_array (np.ndarray): 相对计算时间序列

        Out:
            self.Resv_history_outflow (RainData): 插补后的历史泄洪过程字典

        """
        Resv_history_outflow = {}
        with open(self.Resv_histfile,'r',encoding="utf-8") as fr:
            js = json.load(fr)['discharge']
            for item in js:
                for r in item:
                    Resv_history_outflow[r] = [[],[]]
                    for info in item[r]:
                        Resv_history_outflow[r][0].append(info['time'])
                        Resv_history_outflow[r][1].append(info['value'])
        
        header = ['time','outflow']
        current_DT = datetime.strptime( self.time,self.tform )
        self.Resv_history_outflow = {}
        fill_time_array = self.time_array[:self.currentT_index] #需填补的时间序列
        fill_relT_array = self.relT_array[:self.currentT_index] #需填补的相对时间序列
        for r in self.Reservoir:
            file_time_array = Resv_history_outflow[r][0]
            file_outflow    = Resv_history_outflow[r][1]

            #整理历史泄洪流量
            file_relT_array = np.zeros(len(file_time_array))
            for i,t in enumerate(file_time_array):
                file_relT_array[i] = (
                    datetime.strptime(t,self.tform).timestamp() - current_DT.timestamp()
                )/3600/self.Hydro_Dt
            data = np.column_stack((
                file_time_array,file_outflow
            ))
            np.savetxt(f"{self.pathdict['整理结果']}/history_{r}.csv",
                       data,delimiter=',',fmt="%s",encoding="utf-8",
                       header=','.join(item for item in header),comments='')
            
            #如果给定的历史泄洪流量的最近记录时刻离当前时刻很远，插值结果可能完全不对
            if file_relT_array[-1] <= -12: 
                if self.log: print(f"{YELLOW}Warning: the last time of Resv {r} history outflow may be too far from now!{RESET}")
                if self.log: print(f"{YELLOW}         the interpolation may be totally wrong!{RESET}")

            #插值填补到计算时间序列上
            f = interp1d( file_relT_array, Resv_history_outflow[r][1],
                          kind='linear', fill_value='extrapolate' )

            self.Resv_history_outflow[r] = f(fill_relT_array)
            #超出给定历史泄洪数据范围的插值结果将被赋予数据边缘的值
            for i,relT in enumerate(fill_relT_array):
                if relT < file_relT_array[0]:  self.Resv_history_outflow[r][i] = file_outflow[0]
                if relT > file_relT_array[-1]: self.Resv_history_outflow[r][i] = file_outflow[-1]

        if self.log: print(f"{BLUE}Resv history outflow:{RESET}")
        for r in self.Resv_history_outflow:
            if self.log: print(f"    {r}({len(self.Resv_history_outflow[r])}):{self.Resv_history_outflow[r]}")

        #TODO 日计算下用历史泄洪流量文件填补（history）时，计算历史日（平均）泄洪流量


    def Generate_PRCP_TS_station(self,pP2aP:str,
                                 output:Tuple[str,None]=None,
                                 debug:bool=False):
        """通过站点点雨量生成流域面雨量文件

        Args:
            pP2aP (str):
            output (str|None):
            debug (bool):
        
        In:
            self.settings (dict): 计算设置
            self.RData (RainData): 点位雨量数据
            self.pathdict (dict): 文件路径字典
            self.time (str): 计算时刻，年月日时分秒 
            self.tform (dict): 年月日时分秒

        """
        RData = self.RData
        if output is None:
            output = self.pathdict['模型输入']
        if RData.EPSG != self.settings['EPSG']:
            RData.points = Transform_CRS(RData.points,
                                         RData.EPSG,self.settings['EPSG'],
                                         output=None)

        # ======================================================================
        #                              固定依赖文件                             
        # ======================================================================
        sub_basinshp  = self.settings['子流域划分文件']
        voronoi_shp   = self.settings['泰森多边形文件']

        subB_name_field = self.settings['子流域名称字段']
        subB_sort_field = self.settings['子流域排序字段']
        subB_area_field = self.settings['子流域面积字段']

        voro_name_field = self.settings['雨量测站名称字段']
        voro_sort_field = self.settings['雨量测站排序字段']

        if (self.input_type=='station') and (self.pP2aP=='IDW(mesh)'):
            gridshp       = self.settings['网格文件']
            basinshp      = self.settings['流域范围文件']
            choiceshp     = self.settings['雨量测站选取文件']

        names_basin_sorted = ReadTable( sub_basinshp,
                                        subB_name_field,subB_sort_field)[0]

        # ======================================================================
        #                                计算                             
        # ======================================================================
        #泰森多边形得到的子流域面雨量
        if pP2aP == 'voro':
            arearain_file  = f"{output}/PRCP_voro.csv"
            ratio = Calc_Voronoi_ratio( voronoi_shp, voro_name_field,
                                        sub_basinshp,subB_name_field,
                                        sort_voro=voro_sort_field,
                                        sort_subbasin=subB_sort_field,
                                        output=None )
            basinarearain = np.dot(ratio,RData.rains)
            basinarearain = np.round(basinarearain,2)
        #反距离网格化插值得到的子流域面雨量
        elif pP2aP == 'IDW(mesh)':
            arearain_file = f"{output}/PRCP_IDW.csv"
            basinarearain = Calc_PRCP_IDW(self.mesh_parafile,RData,
                                          gridshp,basinshp,choiceshp,
                                          voronoi_shp,voro_name_field,
                                          sub_basinshp,subB_name_field,subB_sort_field,
                                          debug)
        else:
            raise ValueError(f"pP2aP {pP2aP} is not supported, use 'IDW(mesh)' or 'voro'.")

        #计算流域平均面雨量
        avrgarearain_file = f"{output}/PRCP_avrg.csv"
        subbasin_area  = ReadTable(sub_basinshp,subB_area_field)
        basin_arearain_voro = np.dot(basinarearain.T,subbasin_area)/np.sum(subbasin_area)
        basin_arearain_voro = np.round(basin_arearain_voro,2)


        # ======================================================================
        #                                输出                             
        # ======================================================================
        #子流域面雨量文件
        header = ['时间'] + names_basin_sorted 
        data = np.column_stack(( RData.Tseries, basinarearain.T))
        np.savetxt(arearain_file,data,delimiter=",",fmt='%s',
                    header=','.join(item for item in header),
                    comments='',encoding="utf-8")
        
        #流域平均面雨量文件
        header = ['时间','流域平均']
        data = np.column_stack(( RData.Tseries, basin_arearain_voro ))
        np.savetxt(avrgarearain_file,data,delimiter=",",fmt='%s',
                    header=','.join(item for item in header),
                    comments='',encoding="utf-8")

        if self.log: print(f"{BLUE}create{RESET} {arearain_file}")
        if self.log: print(f"{BLUE}create{RESET} {avrgarearain_file}")


    def Generate_PRCP_TS_mesh(self,output=None,debug=False):
        """通过网格点雨量生成流域面雨量文件

        In:
            self.settings (dict): 计算设置
            self.RData (RainData): 点位雨量数据
            self.pathdict (dict): 文件路径字典
            self.time (str): 计算时刻，年月日时分秒 
            self.tform (dict): 年月日时分秒

        """
        RData = self.RData
        if output is None:
            output = self.pathdict['模型输入']
        if RData.EPSG != self.settings['EPSG']:
            RData.points = Transform_CRS(RData.points,
                                         RData.EPSG,self.settings['EPSG'],
                                         output=None)

        # ======================================================================
        #                              固定依赖文件                             
        # ======================================================================
        gridshp       = self.settings['网格文件']
        basinshp      = self.settings['流域范围文件']
        sub_basinshp  = self.settings['子流域划分文件']

        subB_name_field = self.settings['子流域名称字段']
        subB_sort_field = self.settings['子流域排序字段']
        subB_area_field = self.settings['子流域排序字段']

        # ======================================================================
        #                         定义生成的模型输入文件                             
        # ======================================================================
        output_mrain  = f"{output}/PRCP_mesh.csv"
        output_hmrain = f"{output}/PRCP_avrg.csv"

        # ======================================================================
        #                             网格面雨量计算                              
        # ======================================================================
        arearain = AreaRain_anlysis(gridshp,basinshp,RData,debug=debug)[1] #统一的网格面雨量

        BasinPolys = ReadPolys(sub_basinshp)[0]
        basinarearain = np.zeros( (len(BasinPolys),RData.num_time) )
        for iter,basin_poly in enumerate(BasinPolys):
            coverarea = Intersection_analysis(gridshp,basin_poly,debug=debug)
            basinarearain[iter] = Calc_BasinAreaRain(arearain,coverarea)
        if debug:
            new_Field = {'arearain':basinarearain[:,0]}
            Add_field(sub_basinshp,new_Field)

        names_basin_sorted,order = ReadTable(sub_basinshp,subB_name_field,subB_sort_field)
        basinarearain = basinarearain[order]

        # ======================================================================
        #                         流域平均面雨量                        
        # ======================================================================
        #根据泰森多边形子流域面雨量，计算流域平均面雨量
        subbasin_area  = ReadTable(sub_basinshp,subB_area_field)
        basin_arearain_voro = np.dot(basinarearain.T,subbasin_area)/np.sum(subbasin_area)
        basin_arearain_voro = np.round(basin_arearain_voro,2)

        # ======================================================================
        #                         输出流域面雨量文件                        
        # ======================================================================
        #网格化降雨文件
        header = ['时间'] + names_basin_sorted 
        data = np.column_stack(( RData.Tseries, basinarearain.T))
        np.savetxt(output_mrain,data,delimiter=",",fmt='%s',
                    header=','.join(item for item in header),
                    comments='',encoding="utf-8")
        
        #流域平均面雨量
        header = ['时间','流域平均']
        data = np.column_stack(( RData.Tseries, basin_arearain_voro ))
        np.savetxt(output_hmrain,data,delimiter=",",fmt='%s',
                    header=','.join(item for item in header),
                    comments='',encoding="utf-8")
        
        if self.log: print(f"{BLUE}create{RESET} {output_mrain}")
        if self.log: print(f"{BLUE}create{RESET} {output_hmrain}")

    def Generate_PRCP_TS(self,output:Tuple[str,None]=None,debug:bool=False):
        if output is not None:
            if not os.path.exists(output):
                os.makedirs(output,exist_ok=True)

        if self.input_type == "station":
            self.Generate_PRCP_TS_station(self.pP2aP,output,debug)
        elif self.input_type == "mesh":
            self.Generate_PRCP_TS_mesh(output,debug)

    def Generate_EVAP_TS(self):
        """生成蒸发文件

        In:
            self.time_array (dict): 计算设置
            self.num_subbasins (RainData): 点位雨量数据
            self.pathdict (dict): 文件路径字典
            self.mode (str): 日计算还是次洪计算
            self.tform (dict): 年月日时分秒

        """
        E_data = np.ones(( len(self.time_array), self.num_subbasins))
        for i,t in enumerate(self.time_array):
            dt = datetime.strptime(t, self.tform)
            if self.mode == 'flood':
                E_data[i] = self.EinMonth[dt.month] /30/24
            elif self.mode == 'day':
                E_data[i] = self.EinMonth[dt.month] /30  
        E_data = np.round(E_data,3)
        header = ['时间'] + self.names_subbasin
        output_hmpet = f"{self.pathdict['模型输入']}/EVAP_avrg.csv"
        data = np.column_stack(( self.time_array, E_data ))
        np.savetxt(output_hmpet,data,delimiter=",",fmt='%s',
                    header=','.join(item for item in header),
                    comments='',encoding="utf-8")

    def Generate_addition_data(self,input_file,type):
        """判断输入文件是否覆盖计算时段，若起始时刻不在则报错；若终止时刻不在则按情况填补。

        Args:
            input_file (str): 输入文件中不存在计算时段中第一个时刻的值
            type (srt): 降雨输入文件还是蒸发输入文件，"PRCP"或"EVAP"

        Raises:
            ValueError: 输入文件中不存在计算时段中第一个时刻的值
            ValueError: type类型不支持

        Returns:
            Tuple: [起始时刻在输入文件中的索引,终止时刻在输入文件中的索引],\n
                   [填补文件，当前时刻在填补文件中的索引，终止时刻在填补文件中的索引]
        """
        file_info = np.loadtxt(input_file,skiprows=1,delimiter=',',usecols=(0),
                            ndmin=2,encoding="utf-8",dtype=str)
        file_time = file_info[:,0]
        #计算起始时刻在输入文件的位置，计算终止时刻在输入文件的位置，计算时刻在输入文件中的位置
        time_index = [None,None,None] 
        for iter,item in enumerate(file_time):
            if item == self.calc_period_complete[0]:
                time_index[0] = iter
            if item == self.calc_period_complete[1]:
                time_index[1] = iter
            if item == self.time:
                time_index[2] = iter
        if time_index[0] is None: #输入文件中不存在计算时段中第一个时刻的值
            raise ValueError(f"{self.calc_period_complete[0]} not in {input_file}.")
        if time_index[1] is None: #输入文件中不存在计算时段中最后一个时刻的值
            pass
        else: #输入文件中已包含计算时段
            if (type == "EVAP") or (self.settings['降雨补充数据'] == 'auto'):
                if self.log: print(f"{BLUE}no need to geonerate addition data for{RESET} {input_file}.")
                return time_index,[None,None,None]
        
        file_last_DT = datetime.strptime(file_time[-1],self.tform)
        ft_DT = datetime.strptime(self.calc_period_complete[1],self.tform)
        num_add = int( (ft_DT.timestamp()- file_last_DT.timestamp())/(self.Hydro_Dt*3600) )
        add_time = [file_last_DT+timedelta(hours=i*self.Hydro_Dt) for i in range(1,num_add+1)]

        if type == "EVAP":
            add_data = np.zeros(( num_add, self.num_subbasins))
            time_array = []
            for i in range(num_add):
                if self.mode == 'flood':
                    add_data[i] = self.EinMonth[add_time[i].month] /30/24
                elif self.mode == 'day':
                    add_data[i] = self.EinMonth[add_time[i].month] /30
                time_array.append(add_time[i].strftime(self.tform))
            time_array = np.array(time_array)
            add_data = np.round(add_data,3)
            header = ['时间'] + self.names_subbasin
            output = f"{self.pathdict['补充数据']}/add_EVAP.csv"
            data = np.column_stack(( time_array, add_data ))
            np.savetxt(output,data,delimiter=",",fmt='%s',
                        header=','.join(item for item in header),
                        comments='',encoding="utf-8")
            if self.log: print(f"{BLUE}add evaporation data:{RESET} {time_array[0]}~{time_array[-1]}")

        elif type == "PRCP":
            #若降雨补充数据非自动补零
            if self.settings['降雨补充数据'] != 'auto':
                mark = self.settings['降雨补充数据']
                output = f"{self.pathdict['补充数据']}/PRCP_{mark}.csv"
                output_file_time = np.loadtxt(input_file,skiprows=1,delimiter=',',usecols=(0),
                                    encoding="utf-8",dtype=str)
                add_cplus1,add_n = [None,None]
                for iter,item in enumerate(output_file_time):
                    if item == self.time_array[self.currentT_index+1]:
                        add_cplus1 = iter
                    if item == self.calc_period_complete[1]:
                        add_n = iter
                return time_index,[output,add_cplus1,add_n]
            
            #降雨补充数据为自动补零
            add_data = np.zeros(( num_add, self.num_subbasins))
            time_array = []
            for i in range(num_add):
                time_array.append(add_time[i].strftime(self.tform))
            time_array = np.array(time_array)
            
            header = ['时间'] + self.names_subbasin
            output = f"{self.pathdict['补充数据']}/add_PRCP.csv"
            
            data = np.column_stack(( time_array, add_data ))
            np.savetxt(output,data,delimiter=",",fmt='%s',
                        header=','.join(item for item in header),
                        comments='',encoding="utf-8")
            if self.log: print(f"{BLUE}add precipitation data:{RESET} {time_array[0]}~{time_array[-1]}")
        else:
            raise ValueError(f"type {type} not supported!")

        return time_index,[output,None,None]


    def Get_PRCP_EVAP(self,savePRCP:Tuple[str,None]):
        """获取计算时段的降雨蒸发输入

        Args:
            savePRCP (str|None): 保存读取到的降雨输入

        Raises:
            ValueError: 根据起始和终止时刻读取到的输入数据长度和预设的时间序列长度（等时间间隔）不一致
            ValueError: 根据起始和终止时刻读取到的输入数据时间序列与预设的时间序列不一致

        Returns:
            Tuple: 降雨输入数组,蒸发输入数组
        """
        #读取输入降雨蒸发
        if self.input_type == 'station':
            if self.pP2aP == "IDW(mesh)":
                p_file = f"{self.pathdict['模型输入']}/PRCP_IDW.csv"
            elif self.pP2aP == "voro":
                p_file = f"{self.pathdict['模型输入']}/PRCP_voro.csv"
        elif self.input_type == 'mesh':
            p_file = f"{self.pathdict['模型输入']}/PRCP_mesh.csv"

        e_file = f"{self.pathdict['模型输入']}/EVAP_avrg.csv"
        p_data = np.loadtxt(p_file,skiprows=1,delimiter=',',
                            ndmin=2,encoding="utf-8",dtype=str)
        e_data = np.loadtxt(e_file,skiprows=1,delimiter=',',
                            ndmin=2,encoding="utf-8",dtype=str)    
        if self.log: print(f"{BLUE}precipitation{RESET}: {p_file}")
        if self.log: print(f"{BLUE}evaporation{RESET}: {e_file}")

        #补充降雨蒸发数据
        p_time_index,[p_add_file,add_cplus1,add_n] = self.Generate_addition_data(p_file,'PRCP')
        m,n,c = p_time_index
        if p_add_file is not None:
            if self.log: print(f"{BLUE}add precipitation{RESET}: {p_add_file}")
            add_p  = np.loadtxt(p_add_file,skiprows=1,delimiter=',',
                                ndmin=2,encoding="utf-8",dtype=str)
            if self.settings['降雨补充数据'] == 'auto': #降雨补充数据可直接对接
                p_data = np.vstack((p_data[m:,:],add_p))
            else: #对应降雨补充数据不为auto的情况，降雨补充数据不可直接对接，查看所需的数据范围
                if self.log: print(f"    find index {add_cplus1} ~ {add_n}")
                if add_cplus1 is None:
                    print(f"{RED}Error: the time next to current time not in {p_add_file}.{RESET}")
                if add_n is None:
                    print(f"{RED}Error: not enough data in {p_add_file}.{RESET}")
                    p_data = np.vstack((p_data[m:c+1,:],add_p[add_cplus1:,:]))
                else:
                    p_data = np.vstack((p_data[m:c+1,:],add_p[add_cplus1:add_n+1,:]))
        else:
            p_data = p_data[m:n+1,:]

        e_time_index,[e_add_file,add_cplus1,add_n] = self.Generate_addition_data(e_file,'EVAP')
        m,n,c = e_time_index
        if e_add_file is not None:
            if self.log: print(f"{BLUE}add evaporation{RESET}: {e_add_file}")
            add_e  = np.loadtxt(e_add_file,skiprows=1,delimiter=',',
                                ndmin=2,encoding="utf-8",dtype=str)
            e_data = np.vstack((e_data[m:,:],add_e))
        else:
            e_data = e_data[m:n+1,:]
        
        #检查输入的降雨文件时间序列是否符合预期
        if len(self.time_array) != len(p_data[:,0]):
            header = ['时间'] + self.names_subbasin
            np.savetxt("wrong_input_raindata.csv",p_data,delimiter=',',fmt="%s",
                        header=','.join(item for item in header),
                        comments='',encoding='utf-8')
            raise ValueError(
                f"unmatched number of time,  expect {len(self.time_array)} while get {len(p_data[:,0])}."
            )
        for i in range(len(self.time_array)):
            if self.time_array[i] != p_data[i,0]:
                header = ['时间'] + self.names_subbasin
                np.savetxt("wrong_input_raindata.csv",p_data,delimiter=',',fmt="%s",
                           header=','.join(item for item in header),
                           comments='',encoding='utf-8')
                raise ValueError(
                    f"wrong time in input raindata: {p_data[i,0]}.\nsee row{i} in wrong_input_raindata.csv"
                )
            
        if savePRCP is not None:
            header = ['时间'] + self.names_subbasin
            np.savetxt( savePRCP,p_data,delimiter=',',fmt="%s",
                        header=','.join(item for item in header),
                        comments='',encoding='utf-8')
        
        return p_data,e_data


    def Calc(self,savePRCP:Tuple[str,None]=None,debug:bool=False):
        np.set_printoptions(suppress=True)

        # ======================================================================
        #                              降雨蒸发输入                             
        # ======================================================================
        p_data,e_data = self.Get_PRCP_EVAP(savePRCP) 
        
        # ======================================================================
        #                              水文计算                             
        # ======================================================================
        time_array = self.time_array
        relT_array = self.relT_array
        p_data = np.array(p_data[:,1:],dtype=float)
        e_data = np.array(e_data[:,1:],dtype=float)
        #计算并保存计算结果
        Hydro_Q =  Calc_Hydro(
            self.model,
            time_array,p_data,e_data,
            self.Hydro_parafile,self.Hydro_initfile,self.Hydro_Dt,
            self.pathdict['模型输出']
        )
        if self.log: print(f"{BLUE}Hydro calculation completed.{RESET}")


        # ======================================================================
        #                        马斯京根和水库调洪流量初始化                             
        # ======================================================================
        if self.settings['马斯京根'] == 'YES':
            #初始化马斯京根出口断面流量数组
            MSKG_Q = np.zeros((self.num_MSKGs, len(time_array))) * np.nan
        else:
            MSKG_Q = None

        if self.settings['水库调洪'] == 'YES':
            #初始化水库出口流量数组
            Resv_Q = {}
            for r in self.Reservoir:
                Resv_Q[r] = np.zeros(len(time_array)) * np.nan

            #如有，设置自定义泄洪流量
            if self.is_UDF_outflow:
                for r in self.UDF_outflow:
                    Resv_Q[r][self.currentT_index] = self.Resv_init[r][0] #当前时刻出库流量
                    Resv_Q[r][self.currentT_index+1:] = self.UDF_outflow[r] #未来预见期内的出库流量
                if self.log: print(f"{MAGENTA}{r} outflow is given.{RESET}")
            #若历史泄洪流量填补方式为history，则根据历史泄洪流量数据填补；若为auto，则不做处理，在后续自动设置为入库流量
            if self.Resv_fill_way == 'history':
                for r in self.Resv_history_outflow:
                    Resv_Q[r][:self.currentT_index] = self.Resv_history_outflow[r] #历史出库流量，不包括当前时刻
            elif self.Resv_fill_way == 'auto':
                pass
            if debug:
                if self.log: print(f"{MAGENTA}initial outflow:{RESET}")
                for r in self.Reservoir:
                    if self.log: print(f"    {r}:{Resv_Q[r]}")
        else:
            Resv_Q = None


        # ======================================================================
        #                        马斯京根和水库调洪计算                             
        # ======================================================================
        is_continue = (self.settings['马斯京根'] == 'YES') or (self.settings['水库调洪'] == 'YES')
        #判断是否已计算
        is_MSKG_calculated = np.zeros(self.num_MSKGs) > 1 #马斯京根
        is_Resv_calculated = {} #水库调洪
        for r in self.Reservoir:
            is_Resv_calculated[r] = False
        #记录循环次数
        loop = 0
        while is_continue:
            #--------------------------------马斯京根计算----------------------------------------
            if self.settings['马斯京根'] == 'YES':
                to_calc_MSKG = []
                for iMSKG in range(self.num_MSKGs):
                    MSKG_Dt = self.MSKG_para[iMSKG,0]
                    #判断是否已经计算过
                    if is_MSKG_calculated[iMSKG]: continue
                    #计算马斯京根出口断面流量，当输入有NaN时返回None
                    if self.log: print(f"{MAGENTA}calculating{RESET} M{iMSKG}")
                    Q = Calc_MSKG(  iMSKG, MSKG_Q, MSKG_Dt,
                                    self.MSKG_input, self.MSKG_input_loc,
                                    self.MSKG_para_seg, self.MSKG_init,
                                    Hydro_Q, self.Hydro_Dt,
                                    Resv_Q,
                                    debug=debug,log=self.log  )
                    if Q is None:
                        to_calc_MSKG.append("M%d"%iMSKG)
                    else:
                        MSKG_Q[iMSKG] = Q
                        is_MSKG_calculated[iMSKG] = True
                is_MSKG_allok = (to_calc_MSKG == [])
                if (not is_MSKG_allok) and (self.log): print(f"{MAGENTA}to_calc_MSKG: {to_calc_MSKG}{RESET}")
            else:
                is_MSKG_allok = True


            #--------------------------------水库调洪计算----------------------------------------
            if self.settings['水库调洪'] == 'YES':
                to_calc_Resv = []
                for iResv in range(self.num_Resv):
                    Rname = self.names_Resv[iResv]
                    #判断是否已经计算过，若是则不计算
                    if is_Resv_calculated[Rname]: continue
                    if self.log: print(f"{MAGENTA}calculating{RESET} {Rname}")
                    #判断入库流量是否为NaN，当输入有NaN时返回None
                    Resv_qin = np.zeros(len(time_array))
                    for item in self.Resv_input[Rname]:
                        inflow = Get_inflow_from_str(item,MSKG_Q,Hydro_Q,Resv_Q)
                        if True in np.isnan(inflow):
                            Resv_qin = None
                            if self.log: print(f"    {YELLOW}delayed because NaN in {item}{RESET}")
                            break
                        Resv_qin += inflow
                    if Resv_qin is None:
                        to_calc_Resv.append(Rname)
                        continue
                    #如果历史水库泄洪流量为NaN，则按出库等于入库进行填补
                    if True in np.isnan(Resv_Q[Rname][:self.currentT_index+1]):
                        Resv_Q[Rname][:self.currentT_index+1] = Resv_qin[:self.currentT_index+1]
                    if debug and self.log:
                        print(f"    init:{self.Resv_init[Rname]}")
                        print(f"    Qin: {Resv_qin}")
                        print(f"    Qout_initial:{Resv_Q[Rname][:self.currentT_index+1]} | {Resv_Q[Rname][self.currentT_index+1:]}")
                    #计算未来出库流量并保存水库调洪结果
                    output = f"{self.pathdict['模型输出']}/routing_{Rname}.csv"
                    Resv_Q[Rname][self.currentT_index:] = self.Resv_routing[Rname](
                        self.Reservoir[Rname],
                        time_array[self.currentT_index:],
                        Resv_qin[self.currentT_index:],
                        Resv_Q[Rname][self.currentT_index:],
                        self.Resv_init[Rname][0],self.Resv_init[Rname][1],
                        output,epsilon=1.0,
                        logfile=None
                    )
                    if debug and self.log:
                        print(f"    Qout_final:  {Resv_Q[Rname][:self.currentT_index+1]} | {Resv_Q[Rname][self.currentT_index+1:]}")
                    is_Resv_calculated[Rname] = True
                is_Resv_allok = (to_calc_Resv == [])
                if (not is_Resv_allok) and (self.log): print(f"{MAGENTA}to_calc_Resv: {to_calc_Resv}{RESET}")
            else:
                is_Resv_allok = True

            if self.log: print("-"*80)
            if (is_MSKG_allok and is_Resv_allok):
                is_continue = False
            if (loop > 2*len(self.to_calc)): #循环次数远超过要计算的马斯京根出口断面/水库泄洪流量个数
                is_continue = False
                print(f"{YELLOW}break loop since number of loops >> len(to_calc).{RESET}")
                print(f"{YELLOW}please check each model's input.{RESET}")
        # ======================================================================                         
        # ======================================================================

        if self.settings['马斯京根'] == 'YES':
            #保存马斯京根演算结果
            MSKG_Q = np.round(MSKG_Q,2)
            MSKGQdata  = np.column_stack((time_array,relT_array,MSKG_Q.T))
            Header = ['time','relT'] + self.names_MSKG
            np.savetxt( f"{self.pathdict['模型输出']}/Q_MSKG.csv",  MSKGQdata,
                        delimiter=',',fmt="%s",
                        header=",".join(item for item in Header),comments='')
            np.savetxt( f"{self.pathdict['模型输出']}/new_MSKGQs.csv",
                        MSKGQdata[-1,2:].T,  delimiter=',',fmt="%s",
                        header="Q_init",comments='')
            if self.log: print(f"{BLUE}MSKG calculation completed.{RESET}")  
        if self.settings['水库调洪'] == 'YES':
            if self.log: print(f"{BLUE}Resv routing calculation completed.{RESET}")  


    def Save_new_initfile(self):
        newstate_file = f"{self.pathdict['模型输出']}/new_States.csv"
        newQinit_file = f"{self.pathdict['模型输出']}/new_MSKGQs.csv"

        last_time = datetime.strptime(self.time_array[-1], self.tform)
        last_time = last_time.strftime(self.dtform) #年月日

        new_path = f"{self.pathdict['初始状态']}/States_{last_time}.csv"
        if os.path.exists(new_path):
            raise FileExistsError(f"{new_path} already exists.")
        shutil.copy(newstate_file,new_path)
        print('create file %s'%new_path)

        new_path = f"{self.pathdict['初始状态']}/MSKGQs_{last_time}.csv"
        if os.path.exists(new_path):
            raise FileExistsError(f"{new_path} already exists.")
        shutil.copy(newQinit_file,new_path)
        print('create file %s'%new_path)

    
def Parallel_Calc(func:Callable,args_list:list,ncsize:int):
    s = time.time()
    with mp.Pool(ncsize) as pool:
        results = pool.starmap(func,args_list)
    e = time.time()
    print("%.2f s"%(e-s))
    return results

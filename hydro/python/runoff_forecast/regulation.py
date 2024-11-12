import re
import copy
from datetime import datetime
import numpy as np
from scipy.interpolate import interp1d
from typing import IO

from print_colors import *

def OpenLog(logfile:str,mode:str):
    if logfile == None:
        return None
    elif isinstance(logfile,str):
        fw = open(logfile,mode)
        return fw
    else:
        raise ValueError("not a valid input")

def WriteLog(fw:IO,string:str):
    if fw == None:
        pass
    else:
        fw.write(string)
def CloseLog(fw:IO):
    if fw == None:
        pass
    else:
        fw.close()

#根据水量平衡方程计算下一时刻的水库库容
def compute_next_V( qin_t0, qin_t1,
                    qout_t0,qout_t1,
                    V_t0,deltaT):
    V_t1 = ((qin_t0+qin_t1)/2 - (qout_t0+qout_t1)/2)*deltaT*3600 + V_t0
    return V_t1

class Rule():
    def __init__(self,priority,condition,action,functions_name_list,log:bool=False) -> None:
        self.log:bool = log
        self.rule_name = "rule%d"%priority
        self.condition = condition
        self.action    = action
        self.is_a_rulefunc = False
        self.func_name = None

        self.Preprocess(functions_name_list)

    def Preprocess(self,functions_name_list):
        #预处理判断语句字符串
        if '&' in self.condition:
            conds = self.condition.split('&')
            condition_string = '&'.join('(%s)'%item for item in conds)
        elif '|' in self.condition:
            conds = self.condition.split('|')
            condition_string = '|'.join('(%s)'%item for item in conds)
        else:
            conds = [self.condition]
            condition_string = '(%s)'%self.condition
    
        #预处理执行语句字符串
        action_string = self.action.split('=')[-1]

        self.condition = condition_string
        self.action    = action_string
        
        #检验判断和执行语句字符串
        #去除特定字符，有效的判断和执行语句在去除操作后应只剩下数字或调度函数名
        eliminated_items = [
            'Qout','Qin','SL','Pcpn','min','max',
            '(',')','<','>','=','|','&','+','-','*','/','.'
        ]
        new_condition_string = copy.deepcopy(condition_string)
        for item in eliminated_items:
            new_condition_string = new_condition_string.replace(item,'')
        new_action_string = copy.deepcopy(action_string)
        for item in eliminated_items:
            new_action_string = new_action_string.replace(item,'')
        
        #是否匹配到字母字符串
        pattern = ".*([a-zA-Z]+).*"
        reobj = re.match(pattern,new_condition_string)
        if reobj is not None:
            raise ValueError(f"not a valid condition for {self.rule_name}")

        reobj = re.match(pattern,new_action_string)
        is_a_rulefunc = False #判断执行语句是否为调度函数
        if reobj is not None: #匹配到functions名
            if new_action_string not in functions_name_list:
                raise ValueError(
                    f"{new_action_string} is not a valid function for {self.rule_name}"
                )
            else:
                is_a_rulefunc = True
                func_name = new_action_string
                if self.log: print(f"    found function '{func_name}' for {self.rule_name}")
        else:
            func_name = None
        
        self.is_a_rulefunc = is_a_rulefunc
        self.func_name = func_name

    def rule_func(self,Qin,SL,Pcpn=None):
        if eval(self.condition,{'Qin':Qin,'SL':SL,'Pcpn':Pcpn}):
            if self.is_a_rulefunc:
                rule_judge_info = self.func_name
            else:
                rule_judge_info = eval(self.action,{'Qin':Qin,'SL':SL,'Pcpn':Pcpn})
        else:
            rule_judge_info = None
        return rule_judge_info

class Reservoir_Reg():
    def __init__(self, definition_file, log:bool=False) -> None:
        self.log:bool = log

        self.NAME = "" #水库名
        self.RULES = {} #读取到的调度规则
        self.FUNCTIONS = {} #读取到的函数
        self.STORAGE_CURVE = np.array([]) #读取到的库容曲线
        self.CURVES = {} #读取到的曲线
        self.POWER_DISC = 0.0 #读取到的发电流量

        self.rule_pyfuncs = {} #调度规则函数
        self.funtions = {} #定义文件内函数的字典
        self.discharge_func = None #泄洪函数
        self.SL2V_func = None #水位转库容函数
        self.V2SL_func = None #库容转水位函数
        self.curves_dict = {} #曲线字典

        self.num_rules = 0 #调度函数数目
        self.storage_magnitude = 0 #库容量级
        self.rules_priority_list = [] 
        self.functions_name_list = []
        self.num_curve = 0
        self.curve_list = []

        self.Read_definition(definition_file)


    def Set_discharge_func(self,func_name):
        self.discharge_func = self.funtions[func_name]


    
    def Create_func(self,name):
        func_data = self.FUNCTIONS[name]
        f = interp1d(func_data[:,0],func_data[:,1],
                     kind='linear',fill_value='extrapolate')
        return f

    def extract_name(self,lines,NAME_range):
        line = lines[NAME_range[0]+1]
        name = ''.join(line.split())
        self.NAME = name
        if self.log: print('-'*50)
        if self.log: print(f'NAME: {name}')


    def extract_funcs(self,lines,FUNCTIONS_range):
        #提取函数信息
        if self.log: print('FUNCTIONS extrating')
        func_name_lines = [FUNCTIONS_range[0]+1]
        func_name_list  = []
        func_name_lines.append(FUNCTIONS_range[1])
        for i in range(FUNCTIONS_range[0]+1,FUNCTIONS_range[1]):
            # if self.log: print(lines[i],end='')
            if i in func_name_lines[:-1]:
                name = ''.join(lines[i].split()).split('=')[0] #先去除空格，再通过=号划分
                func_name_list.append(name)
                start_line = i+1
                end_line   = func_name_lines[func_name_lines.index(i)+1]
                func_data = []
                for j in range(start_line,end_line):
                    content = ''.join(lines[j].split())
                    if content == '':
                        break
                    func_data.append(content.split(','))
                func_data = np.array(func_data,dtype=float)
                self.FUNCTIONS[name] = func_data
                self.funtions[name] = self.Create_func(name)
        self.functions_name_list = func_name_list

    def extract_storage_curve(self,lines,STORAGE_CURVE_range):
        #提取库容曲线信息
        if self.log: print('STORAGE CURVE extrating')
        curve_data = []
        for i in range(STORAGE_CURVE_range[0]+1,STORAGE_CURVE_range[1]):
            # if self.log: print(lines[i],end='')
            content = ''.join(lines[i].split())
            if content == '':
                break
            if i == STORAGE_CURVE_range[0]+1:
                text_format = ".*V\(([0-9e+]+)\)"
                magnitude = re.match(text_format,lines[i]).group(1)
                magnitude = float(magnitude)
                self.storage_magnitude = magnitude
                # if self.log: print(magnitude)
            else:
                curve_data.append(content.split(','))
        curve_data = np.array(curve_data,dtype=float)
        curve_data[:,1] *= magnitude
        self.STORAGE_CURVE = curve_data
        self.V2SL_func = interp1d(curve_data[:,1],curve_data[:,0],
                     kind='linear',fill_value='extrapolate')
        self.SL2V_func = interp1d(curve_data[:,0],curve_data[:,1],
                     kind='linear',fill_value='extrapolate')
        if self.log: print("    %s magnitude:%d"%('STORAGE CURVE',magnitude))
        
    def extract_curves(self,lines,CURVE_range):
        if CURVE_range[0] == -999:
            return
        #提取曲线位置和名称信息
        if self.log: print('CURVE extrating')
        CURVE_location = []
        CURVE_identity = []
        for i in range(CURVE_range[0]+1,CURVE_range[1]):
            if 'CURVE' in lines[i]:
                CURVE_location.append(i)
                content = ''.join(lines[i].split())
                CURVE_identity.append(content.split('-')[1])
        CURVE_location.append( CURVE_range[1] )
        # if self.log: print(CURVE_location)

        #对每一条曲线的数据进行提取    
        for iter in range(len(CURVE_location)-1):
            curve_data = []
            for i in range(CURVE_location[iter]+1,CURVE_location[iter+1]):
                # if self.log: print(lines[i],end='')
                content = ''.join(lines[i].split())
                if content == '':
                    break
                if i == CURVE_location[iter]+1:
                    text_format = ".*\(([0-9e+]+)\)" #括号内的为量级
                    try:
                        magnitude = re.match(text_format,lines[i]).group(1)
                        magnitude = float(magnitude)
                    except:
                        magnitude = 1.0 #如果没有找到量级信息或读取错误
                else:
                    curve_data.append(content.split(','))
            curve_data = np.array(curve_data,dtype=str)
            self.CURVES[CURVE_identity[iter]] = curve_data
            self.curves_dict[CURVE_identity[iter]] = {}
            for i,item in enumerate(curve_data[:,0]):
                self.curves_dict[CURVE_identity[iter]][item] = float(curve_data[i,1]) * magnitude
            
            if self.log: print("    %s magnitude:%d"%(CURVE_identity[iter],magnitude))


    def extract_rules(self,lines,RULES_range):
        #提取调度规则信息
        if self.log: print('RULES extrating')
        text_format = "(\d+)IF\((.+)\):(.+)"

        num_rules = 0
        for i in range(RULES_range[0]+1,RULES_range[1]):
            # if self.log: print(lines[i],end='')
            content = ''.join(lines[i].split())
            if content == '':
                break
            rm = re.match(text_format,content)
            priority   = int(rm.group(1))
            condition  = rm.group(2)
            action     = rm.group(3)
            self.RULES[priority] = Rule(priority,condition,action,
                                        self.functions_name_list,
                                        log=self.log)
            self.rule_pyfuncs[priority] = self.RULES[priority].rule_func
            num_rules += 1

        self.num_rules = num_rules
        self.rules_priority_list = list(self.RULES.keys())
        self.rules_priority_list.sort() #按从小到大优先级排列

    def extract_power_disc(self,lines,POWER_DISC_range):
        if self.log: print('POWER DISCHARGE extrating')
        i = POWER_DISC_range[0]+1
        content = ''.join(lines[i].split())
        self.POWER_DISC = float(content)

    def Read_definition(self,definition_file):

        with open(definition_file,'r') as fr:
            lines = fr.readlines()
        order = np.ones(6,dtype=int) * (-999) #定义文件中的排列顺序
        NAME_range = []
        RULES_range = []
        POWER_DISC_range = []
        FUNCTIONS_range = []
        STORAGE_CURVE_range = []
        CURVE_range = []
        for i,line in enumerate(lines):
            if '#NAME' in line:
                order[0] = i
                NAME_range.append(i)
            if '#REGULATION RULES' in line:
                order[1] = i
                RULES_range.append(i)
            if '#POWER DISCHARGE' in line:
                order[2] = i
                POWER_DISC_range.append(i)
            if '#FUNCTIONS' in line:
                order[3] = i
                FUNCTIONS_range.append(i)
            if '#STORAGE CURVE' in line:
                order[4] = i
                STORAGE_CURVE_range.append(i)
            if '#CURVE' in line:
                order[5] = i
                CURVE_range.append(i)
        RULES_range.append(order[2])
        POWER_DISC_range.append(order[3])
        FUNCTIONS_range.append(order[4])
        if order[5] != -999:
            STORAGE_CURVE_range.append(order[5])
            CURVE_range.append(len(lines))
        else:
            STORAGE_CURVE_range.append(len(lines))
            CURVE_range.append(-999)
        
        self.extract_name(lines,NAME_range)
        self.extract_storage_curve(lines,STORAGE_CURVE_range)
        self.extract_funcs(lines,FUNCTIONS_range)
        self.extract_rules(lines,RULES_range)
        self.extract_curves(lines,CURVE_range)
        self.extract_power_disc(lines,POWER_DISC_range)


    #采用试算法计算下一时刻出库流量
    def Calc_TE_method(self,qin_t0, qin_t1,
                       qout_t0,SL_t0,
                       deltaT,epsilon=1.0):
        #试算法 trial-and-error method
        #计算按泄流能力曲线下泄时的水位和出库流量
        #deltaT: 入库流量时间间隔，单位h
        #Qin_series: 入库流量时间序列
        #Qout_init: 初始出库流量
        #epsilon: 许可误差

        #TODO 最大迭代次数

        V_t0 = self.SL2V_func(SL_t0)

        qout_t1_final = 0.0
        qout_t1_guess = qout_t0
        converged = False
        iteration = 0
        logstring = ''
        while not converged:
            V_t1 = compute_next_V(qin_t0,qin_t1,
                                  qout_t0,qout_t1_guess,
                                  V_t0,deltaT) #水量平衡方程
            SL_t1 = self.V2SL_func(V_t1)
            qout_t1 = self.discharge_func(SL_t1)
            diff = abs(qout_t1 - qout_t1_guess)

            iteration += 1
            logstring += '    iter-%d Qt1_guess:%.2f Qt1:%.2f(SLt1=%.3f) | diff=%.2f/%.2f\n'%(
                iteration,qout_t1_guess,qout_t1,SL_t1,diff,epsilon
            )

            if diff <= epsilon:
                qout_t1_final = qout_t1
                converged = True
            else:
                qout_t1_guess = (qout_t1+qout_t1_guess)/2

        return [qout_t1_final,SL_t1,V_t1],iteration,logstring

    #水库调洪计算(目前仅能支持不那么复杂的规则)
    def Reservoir_Routing(self,
                          time_series,Qin_series,Qout_series,
                          Qout_init,SL_init,
                          output,epsilon=1.0,
                          logfile=None):
        tform = "%Y-%m-%d %H:%M:%S" 

        fw = OpenLog(logfile,'w')

        #初始化
        num_time = len(time_series)
        relT_series  = np.zeros(num_time)
        SL_series    = np.zeros(num_time) #水位数组
        SL_series[0] = SL_init
        V_init       = self.SL2V_func(SL_init) 
        V_series     = np.zeros(num_time) #水库库容数组
        V_series[0]  = V_init
        #如果不给定出库流量，初始化出库流量数组，将Qout_init作为第一个时刻的出库流量
        #如果给定出库流量，出库流量数组已知，Qout_init设置无效
        is_Qout_given = True not in np.isnan(Qout_series)
        # if self.log: print(f"    {self.NAME} Qout_given: {is_Qout_given}")
        if not is_Qout_given: 
            Qout_series    = np.zeros(num_time) #出库流量数组
            Qout_series[0] = Qout_init

        #对时刻进行遍历
        for i in range(num_time-1):
            #计算当前时刻与下一时刻的时间差，单位h
            t0 = time_series[i]
            t0_obj = datetime.strptime(t0,tform)
            t1 = time_series[i+1]
            t1_obj = datetime.strptime(t1,tform)
            deltaT = (t1_obj.timestamp() - t0_obj.timestamp())/3600
            relT_series[i+1] = relT_series[i] + deltaT

            # ======================================================================
            #                               已知物理量                              
            # ======================================================================
            #当前时刻物理量
            qin_t0  = Qin_series[i]
            qout_t0 = Qout_series[i]
            SL_t0   = SL_series[i] 
            V_t0    = V_series[i]
            #下一时刻入库流量
            qin_t1  = Qin_series[i+1]

            # ======================================================================
            #                    如果给定出库流量，直接计算水位和库容
            # ======================================================================
            if is_Qout_given:
                qout_t1 = Qout_series[i+1] #下一时刻出库流量
                WriteLog(fw,'    qout1=%.2f'%qout_t1)
                V_t1 = compute_next_V(  qin_t0,qin_t1,
                                        qout_t0,qout_t1,
                                        V_t0,deltaT) #水量平衡方程
                SL_t1 = self.V2SL_func(V_t1)
                
                #保存下一时刻出库流量、水位、水库库容到数组中
                SL_series[i+1]   = SL_t1
                V_series[i+1]    = V_t1
                
                #跳过后续（规则调度）代码
                continue

            # ======================================================================
            #                    根据规则计算出库流量，再计算水位和库容
            #                           确定当前时刻的调度规则                              
            # ======================================================================
            #对调度规则进行循环，确定最终的调度行动，并依次给出调度规则的判断结果rule_judge_info。
            #当rule_judge_info为None时，则意味着该调度规则的判定条件不满足；
            #当rule_judge_info类型为数值时，则意味着找到调度规则，并直接可以给出下一时刻出库流量；
            #当rule_judge_info类型为字符串时，则意味着找到调度规则，但需要通过迭代方式计算下一时刻物理量。
            rule_found = False
            for irule in self.rules_priority_list:
                rule_judge_info = self.rule_pyfuncs[irule](qin_t1, round(SL_t0,2)) #判定时水位保留2位小数，这样水位的等于判定条件可能会更稳定一些
                # WriteLog(fw,irule,rule_judge_info)
                #第一个满足调度判定条件的规则为当前时刻的调度规则
                if rule_judge_info is not None: #当rule_judge_info不为None时，即找到调度规则
                    WriteLog(fw,'time:%d 【qin1=%.2f,SL0=%.2f】 rule%s\n'%(i,qin_t1,SL_t0,irule))
                    rule_found = True
                    break

            #如果没有合适的调度规则，则跳出对时刻的遍历
            if not rule_found:
                WriteLog(fw,'time:%d no rule available for qin1=%d,SL0=%.2f\n'%(i,qin_t1,SL_t0))
                break

            # ======================================================================
            #                               计算下一时刻物理量                              
            # ======================================================================
            #在找到合适的调度规则的情况下，进行下面步骤
            #根据调度规则不能直接确定出库流量（按泄洪曲线下泄），则通过试算法迭代计算下一时刻的出库流量、水位、水库库容
            if isinstance(rule_judge_info, str):
                func_name = rule_judge_info
                self.Set_discharge_func(func_name) #设置泄洪曲线，用于下面试算法计算下一时刻物理量
                WriteLog(fw,"    Set func '%s' as the discharge relationship Qout = f(Z)!\n"%func_name)
                var_t1,iteration,logstring = self.Calc_TE_method(qin_t0,qin_t1,
                                                                 qout_t0,SL_t0,
                                                                 deltaT,epsilon)
                qout_t1,SL_t1,V_t1 = var_t1
                WriteLog(fw,logstring)
                WriteLog(fw,'    %s   iterations %d\n'%(
                    "routing with discharge curve",iteration
                ))
                WriteLog(fw,'    qout1=%.2f\n'%qout_t1)
            #根据调度规则可直接确定出库流量，进而确定下一时刻水库库容和水位
            else:
                qout_t1 = rule_judge_info
                WriteLog(fw,'    qout1=%.2f\n'%qout_t1)
                V_t1 = compute_next_V(qin_t0,qin_t1,
                                      qout_t0,qout_t1,
                                      V_t0,deltaT) #水量平衡方程
                SL_t1 = self.V2SL_func(V_t1)

            # ======================================================================
            #                               保存下一时刻物理量                              
            # ======================================================================
            #保存下一时刻出库流量、水位、水库库容到数组中
            Qout_series[i+1] = qout_t1
            SL_series[i+1]   = SL_t1
            V_series[i+1]    = V_t1
        
        #输出调洪计算结果
        header = ['time','relT','qin','qout','SL','V']
        data = np.column_stack((
            time_series,
            relT_series,
            np.round(Qin_series,2),
            np.round(Qout_series,2),
            np.round(SL_series,2),
            np.round(V_series/self.storage_magnitude,3)
        ))
        np.savetxt(output,data,delimiter=',',fmt='%s',
                   header=','.join(header),comments='')

        CloseLog(fw)

        return np.array(data[:,3],dtype=float)


import re
import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple,Union

from hydromodel.xaj import calc_xaj
from hydromodel.shb import calc_shb
from hydromodel.double_excess_v2 import calc_double_excess
from hydromodel.xaj_vmrm import calc_vmrm

from print_colors import *


Hydro_funcs = {
    "xaj":  calc_xaj,
    "shb":  calc_shb,
    "dbe":  calc_double_excess,
    "vmrm": calc_vmrm
}
Hydro_states = {
    "xaj":  "wu0,wl0,wd0,s0,fr0,qi0,qg0,qt0",
    "shb":  "q0,w0",
    "dbe":  "S1,S2,S3,W1,W2,W3",
    "vmrm": "wu0,wl0,wd0,s0,fr0,qi0,qg0,qt0"
}


def Calc_Hydro( model:str,
                time_array:np.ndarray, p_data:np.ndarray, e_data:np.ndarray, 
                Hydro_parafile:str, Hydro_statefile:str, Hydro_Dt:float,
                output:str):

    #输出
    basinQ_file     = f"{output}/Q_basin.csv" #各子流域出口断面流量过程
    newstate_file   = f"{output}/new_States.csv" #各子流域最后一个时刻的状态文件

    #模型参数
    Hydro_info = np.loadtxt(Hydro_parafile,skiprows=1,delimiter=',',dtype=str)
    basinnames = Hydro_info[:,0]
    Area_list      = np.array(Hydro_info[:,1],dtype=float) * (1e+6)
    Parameter_list = np.array(Hydro_info[:,2:],dtype=float)
    num_basins = len(basinnames)

    #初始状态
    States_list = np.loadtxt(Hydro_statefile,skiprows=1,delimiter=',',ndmin=2)

    #计算
    Hydro_Q,newstate_list = [],[]
    for i in range(num_basins):
        Area      = Area_list[i]
        Parameter = Parameter_list[i]
        States    = States_list[i]

        calc = Hydro_funcs[model]
        Q,newstate = calc(
            p_data[:,i],e_data[:,i],
            Parameter,States,
            Area,Hydro_Dt
        )

        Hydro_Q.append(Q)
        newstate_list.append(newstate[0])
    Hydro_Q = np.array(Hydro_Q)
    Hydro_Q = np.round(Hydro_Q,2) 
    newstate_list = np.array(newstate_list)

    #保存水文计算结果
    relT_array = np.arange(0,len(time_array))*Hydro_Dt
    HydroQdata  = np.column_stack((time_array,relT_array,Hydro_Q.T))
    Header = ['time','relT'] + list(basinnames)
    np.savetxt(basinQ_file, HydroQdata, delimiter=',',fmt="%s",
               header=",".join(item for item in Header),
               comments='',encoding="utf-8")
    np.savetxt(newstate_file, newstate_list, delimiter=',',fmt="%.4f",
               header=Hydro_states[model],
               comments='',encoding="utf-8")
    
    return Hydro_Q





#单段马法函数
#c 为马斯京根的模型参数，由c0,c1和c2三个参数组成，其中c0,c1,c2参数需要满足c0+c1+c2=1的关系
#q_up表示输入流量，为一个时间序列流量，时间间隔应为DeltaT
#q_ini为河道基流，是一个数
def river_c(c, q_up, q_ini):
    c0, c1, c2 = c
    # print(c0,c1,c2)
    q = np.zeros(len(q_up))
    q[0] = q_ini
    for i in range(1, q.size):
        q[i] = c0 * q_up[i] + c1 * q_up[i - 1] + c2 * q[i - 1]
    return q

#多段马法函数
#多段马法中，调用了多次单段马法，实际上是将一个入流演进到下一个入流的位置，将二者相加，再继续演进，但最后只输出一个终点断面的流量过程
#多段马法函数函数中假设每段的单段马法使用的马斯京根参数相同，记为para
#q_in为一个数组，每行表示一个入流过程，该入流过程可以来自水文模型，也可以是来自水库泄洪方案
#q_ini与单段马法相同，为河道基流，为一个值
#输出最后一个断面的流量过程
def muskingum_func(para, q_in, q_ini):
    river_number = q_in.shape[0]
    q_sim = [] * river_number
    if river_number == 1:
        q_up = q_in[0, :]
        Q_sim = river_c(para, q_up, q_ini)
        return np.array(Q_sim).reshape(1, -1)

    for i in range(river_number):
        if i == 0:
            q_up = q_in[0]
            q_sim.append(river_c(para, q_up, q_ini))
        else:
            q_up = np.sum([q_sim[i - 1], q_in[i]], axis=0)
            q_sim.append(river_c(para, q_up, q_ini))
    return np.array(q_sim[river_number - 1]).reshape(1, -1)


# 根据字符串找入流
def Get_inflow_from_str(instr:str,
                        MSKG_Q:np.ndarray,Hydro_Q:np.ndarray,
                        Resv_Q:dict) -> Tuple[np.ndarray,None]:
    #空字符返回零数组
    if instr == '':
        return np.zeros(len(Hydro_Q[0]))
    
    rm = re.match("([M]|[H])([0-9]+)",instr)
    #匹配马斯京根流量过程或水文流量过程
    if rm is not None:
        index = int(rm.group(2))
        if rm.group(1) == 'H': return Hydro_Q[index]
        if rm.group(1) == 'M': return MSKG_Q[index]
    #匹配水库出库流量过程
    else:
        return Resv_Q[instr]   



def Convert_seglength_to_x(MSKG_input_leng):
    MSKG_input_loc = []
    for i,item in enumerate(MSKG_input_leng):
        MSKG_input_loc.append(np.zeros(len(item),dtype=int))
        for j in range(len(item)):
            upstream_station = np.array(MSKG_input_leng[i][0:j+1],dtype=int)
            MSKG_input_loc[i][j] =  np.sum(upstream_station)
    return MSKG_input_loc

#传参包含n时，输出为分段马斯京根参数计算得到的C0、C1、C2、n
def Convert_MSKGpara_to_CCC(MSKGpara):
    '''
    #DeltaT,KE,XE,n\n
    #DeltaT 马斯京根预报流量时间基段\n
    #KE     （总）河段蓄量常数——稳定流流量的传播时间\n
    #XE     （总）河段流量比重系数——反映河槽的调节能力，滞后坦化\n
    #n       河段分段数，使分段蓄量常数基本等于DeltaT\n
    MSKG_oripara = np.array([
        [1, 9, 0.4, 9],
        [1, 4, 0.48, 4],
        [1, 2, 0.48, 2],
        [1, 2, 0.49, 2],
        [1, 2, 0.49, 2]
    ])
    '''

    num_MSKG = len(MSKGpara)
    
    if MSKGpara.shape[1] == 3:
        CCC = np.zeros((num_MSKG,3))
        for i in range(num_MSKG):
            DeltaT,KE,XE = MSKGpara[i]
            denominator = KE - KE*XE + 0.5*DeltaT
            CCC[i][0] = (0.5*DeltaT - KE*XE)/denominator
            CCC[i][1] = (0.5*DeltaT + KE*XE)/denominator
            CCC[i][2] = (KE - KE*XE - 0.5*DeltaT)/denominator
        return CCC
    if MSKGpara.shape[1] == 4:
        CCCn = np.zeros((num_MSKG,4))
        for i in range(num_MSKG):
            DeltaT,KE,XE,n = MSKGpara[i]

            KE_seg = KE/n
            XE_seg = (1-n*(1-2*XE))/2
            denominator = KE_seg - KE_seg*XE_seg + 0.5*DeltaT
            CCCn[i][0] = (0.5*DeltaT - KE_seg*XE_seg)/denominator
            CCCn[i][1] = (0.5*DeltaT + KE_seg*XE_seg)/denominator
            CCCn[i][2] = (KE_seg - KE_seg*XE_seg - 0.5*DeltaT)/denominator
            CCCn[i][3] = int(n)
        return CCCn
        

#等距分段的多段马法
#para_seg: 分段的每段马斯京根ccc参数
#q_in: 马斯京根区间入流
#q_loc: 入流口的里程，需按从小到大排列
def muskingum_func_seg(para_seg, q_in, q_loc, q_init):
    
    n = int(para_seg[3]) #河道分段数
    q_sim = [] * n #每个河段的出流

    #判断q_loc是否按从小到大排列
    for i in range(1,len(q_loc)):
        if q_loc[i] < q_loc[i-1]:
            print("Error: q_loc must be in order, from smallest to largest!")
            break

    seg = np.linspace(0,q_loc[-1],n+1) #将整条河道分成n段后每个节点的里程，n段有n+1个节点
    #print("区间入流处里程 ",q_loc)
    #print("该马斯京根模型全河长%d，分成%d段，每段的分界点为%s"%(q_loc[-1],n,seg)) 
    for i in range(n): #对n个河段进行循环
        #print("-"*20)
        #print("第%d河段 %.1f~%.1f"%(i,seg[i],seg[i+1]))

        #根据侧入流里程x，判断该河段上是否有侧入流
        local_inflow_index = [] #该河段内的所有侧入流
        for index,x in enumerate(q_loc):
            if (x>seg[i]) and (x<=seg[i+1]): #侧入流里程在该河段范围内
                local_inflow_index.append(index) 
        
        if i == 0:
            q_up = q_in[0] #对于第一个河段，上游入流为第一个入流
            q_seg_mskg = river_c(para_seg[0:3], q_up, q_init) #马斯京根演算
            if len(local_inflow_index) :
                summation = np.vstack((q_seg_mskg, q_in[local_inflow_index]))
                q_sim.append( np.sum(summation, axis=0) ) #该河段出流
            else:
                q_sim.append( q_seg_mskg ) #该河段出流
        else:
            q_up = q_sim[i - 1] #对于后续河段，上游入流为经过马斯京根演算后的上一个河段的出流
            q_seg_mskg = river_c(para_seg[0:3], q_up, q_init) #马斯京根演算
            if len(local_inflow_index) :
                summation = np.vstack((q_seg_mskg, q_in[local_inflow_index]))
                q_sim.append( np.sum(summation, axis=0) ) #该河段出流
            else:
                q_sim.append( q_seg_mskg ) #该河段出流
        #print(q_sim[i])

    return np.array(q_sim[-1]).reshape(1, -1)




def Calc_MSKG(iMSKG:int, MSKG_Q:np.ndarray, MSKG_Dt:float,
              MSKG_input:list, MSKG_input_loc:list,
              MSKG_para:np.ndarray, MSKG_q_init:np.ndarray,
              Hydro_Q:np.ndarray, Hydro_Dt:float,
              Resv_Q:dict,
              debug:bool=False,log:bool=False):
    
    if Hydro_Dt != MSKG_Dt:
        if log: print(f"{YELLOW}Warning:{RESET} Hydro_Dt{Hydro_Dt} != MSKG_Dt{MSKG_Dt}")
        Hydro_relT = np.arange(0, Hydro_Q.shape[1], 1) * Hydro_Dt #水文模型的时间序列
        MSKG_relT  = np.arange(0, Hydro_relT[-1]+Hydro_Dt, MSKG_Dt) #以水文模型的时间段，马斯京根的时间间隔重构时间序列
        MSKG_num_time = len(MSKG_relT)

        #水文结果重构
        new_Hydro_Q = np.zeros((Hydro_Q.shape[0], MSKG_num_time))
        for i in range( Hydro_Q.shape[0] ):
            f = interp1d(Hydro_relT,Hydro_Q[i], kind='linear')
            new_Hydro_Q[i] = f(MSKG_relT)

        #马斯京根结果重构
        new_MSKG_Q = np.zeros((MSKG_Q.shape[0], MSKG_num_time))
        for i in range( MSKG_Q.shape[0] ):
            f = interp1d(Hydro_relT,MSKG_Q[i], kind='linear')
            new_MSKG_Q[i] = f(MSKG_relT)

        #水库出库流量重构
        new_Resv_Q = dict()
        for key in Resv_Q:
            if True in np.isnan(Resv_Q[key]):
                new_Resv_Q[key] = np.ones((Hydro_Q.shape[0], MSKG_num_time)) * np.nan
            else:
                f = interp1d(Hydro_relT,Resv_Q[i],kind='linear')
                new_Resv_Q[key] = f(MSKG_relT)
    else:
        new_Hydro_Q = Hydro_Q
        new_Resv_Q  = Resv_Q
        MSKG_num_time = Hydro_Q.shape[1]

    #该马斯京根模型的区间入流数组
    MSKG_qin = np.zeros((len(MSKG_input[iMSKG]), MSKG_num_time))
    for i,item in enumerate(MSKG_input[iMSKG]):
        if '+' in item:
            subitemlist = item.split('+')
            for subitem in subitemlist:
                inflow = Get_inflow_from_str(subitem,MSKG_Q,new_Hydro_Q,new_Resv_Q)
                #如果输入中有待求解的流量过程，则返回None
                if True in np.isnan(inflow):
                    if log: print(f"    {YELLOW}delayed because NaN in {subitem}{RESET}")
                    return None
                MSKG_qin[i] += inflow
        else:
            MSKG_qin[i] = Get_inflow_from_str(item,MSKG_Q,new_Hydro_Q,new_Resv_Q)
            #如果输入中有待求解的流量过程，则返回None
            if True in np.isnan(MSKG_qin[i]):
                if log: print(f"    {YELLOW}delayed because NaN in {item}{RESET}")
                return None  
        if debug and log: print("    ",item,MSKG_qin[i])     

    #各马斯京根模型的出口断面流量演算
    Q = muskingum_func_seg( MSKG_para[iMSKG],                  #各马斯京根模型的单河段参数ccc
                            MSKG_qin, MSKG_input_loc[iMSKG],   #各马斯京根模型的区间入流和入流处里程
                            MSKG_q_init[iMSKG])[0]             #马斯京根河段上游初始流量
    
    #将计算完的马斯京根结果重新插值成水动力计算结果的时间间隔
    if Hydro_Dt != MSKG_Dt:
        f = interp1d(MSKG_relT, Q, kind='linear',fill_value='extrapolate')
        new_Q = f(Hydro_relT)
    else:
        new_Q = Q

    return new_Q


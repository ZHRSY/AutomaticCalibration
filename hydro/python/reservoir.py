import numpy as np
import os,sys
from datetime import datetime

from runoff_forecast.mesh_rain import Read_discdata
from runoff_forecast.regulation import *

def XQ_Routing(ResReg:Reservoir_Reg,
               time_series,Qin_series,Qout_series,
               Qout_init,SL_init,
               output,epsilon=1.0,
               logfile=None):

    discharge = ResReg.Reservoir_Routing( 
                                time_series,Qin_series,Qout_series,
                                Qout_init,SL_init,
                                output,epsilon,
                                logfile)
    return discharge


#水库调洪计算
def BTW_Routing(ResReg:Reservoir_Reg,
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
    V_init       = ResReg.SL2V_func(SL_init) 
    V_series     = np.zeros(num_time) #水库库容数组
    V_series[0]  = V_init
    #如果不给定出库流量，初始化出库流量数组，将Qout_init作为第一个时刻的出库流量
    #如果给定出库流量，出库流量数组已知，Qout_init设置无效
    is_Qout_given = True not in np.isnan(Qout_series)
    # print(f"    {ResReg.NAME} Qout_given: {is_Qout_given}")
    if not is_Qout_given: 
        Qout_series    = np.zeros(num_time) #出库流量数组
        Qout_series[0] = Qout_init
    Qout_power_series    = np.zeros(num_time) #发电流量
    Qout_power_series[0] = min(Qout_init,ResReg.POWER_DISC)  

    Qext_industry_series    = np.zeros(num_time) #工业供水流量
    Qext_evaporation_series = np.zeros(num_time) #蒸发流量
    Qext_leakage_series     = np.zeros(num_time) #渗漏流量

    #对时刻进行遍历
    for i in range(num_time-1):
        
        #计算当前时刻与下一时刻的时间差，单位h
        t0 = time_series[i]
        t0_obj = datetime.strptime(t0,tform)
        t1 = time_series[i+1]
        t1_obj = datetime.strptime(t1,tform)
        deltaT = (t1_obj.timestamp() - t0_obj.timestamp())/3600
        relT_series[i+1] = relT_series[i] + deltaT

        #当前时刻物理量
        qin_t0  = Qin_series[i]
        qout_t0 = Qout_series[i]
        SL_t0   = SL_series[i]
        V_t0    = V_series[i]
        #下一时刻入库流量
        qin_t1  = Qin_series[i+1]

        #当前时刻的所属月份
        mon = str(t0_obj.month)
        #根据月份（水位）确定供水量&蒸发量&渗漏量
        Qext_evaporation_series[i] = ResReg.curves_dict['EVAPORATION'][mon]/30/24/3600
        Qext_leakage_series[i]     = ResReg.curves_dict['LEAKAGE'][mon]/30/24/3600
        if SL_t0 <= ResReg.curves_dict['LOWER_LIMITED_SL'][mon]:
            Qext_industry_series[i] = 0.0
        else:
            Qext_industry_series[i] = ResReg.curves_dict['INDUSTRY'][mon]/30/24/3600
        #预先扣除供水流量&蒸发流量&渗漏流量
        qin_t0_noext = qin_t0 #未扣除前的t0时刻流量
        qin_t1_noext = qin_t1 #未扣除前的t1时刻流量
        Qext_sum = (Qext_evaporation_series[i] + Qext_industry_series[i] + Qext_leakage_series[i])
        qin_t0 -= Qext_sum
        qin_t1 -= Qext_sum

        # ======================================================================
        #                    如果给定出库流量，直接计算水位和库容
        # ======================================================================
        if is_Qout_given:
            qout_t1 = Qout_series[i+1] #下一时刻出库流量
            WriteLog(fw,'    qout1=%.2f'%qout_t1)
            V_t1 = compute_next_V(  qin_t0,qin_t1,
                                    qout_t0,qout_t1,
                                    V_t0,deltaT) #水量平衡方程
            SL_t1 = ResReg.V2SL_func(V_t1)
            
            #保存下一时刻出库流量、水位、水库库容到数组中
            SL_series[i+1]   = SL_t1
            V_series[i+1]    = V_t1
            Qout_power_series[i+1] = min(qout_t1,ResReg.POWER_DISC)
            
            #跳过后续（规则调度）代码
            continue

        # ======================================================================
        #                    根据规则计算出库流量，再计算水位和库容
        # ======================================================================
        #对调度规则进行循环，确定最终的调度行动。如果没有合适的调度规则，则跳出遍历
        rule_found = False
        for irule in ResReg.rules_priority_list:
            rule_judge_info = ResReg.rule_pyfuncs[irule](qin_t1,np.round(SL_t0,2)) #判定时水位保留2位小数，这样水位的等于判定条件会更稳定一些
            # WriteLog(fw,irule,rule_judge_info)
            if rule_judge_info is not None:
                WriteLog(fw,'time:%d 【qin1=%.2f,SL0=%.2f】 rule%s'%(i,qin_t1,np.round(SL_t0,2),irule))
                WriteLog(fw,"    Qin1:%.2f Qext_sum:%.2f industry:%.2f evaporation:%.2f leakage:%.2f"%(
                    qin_t1_noext,Qext_sum,
                    Qext_industry_series[i],Qext_evaporation_series[i],Qext_leakage_series[i]
                ))
                rule_found = True
                break
        if not rule_found:
            WriteLog(fw,'time:%d no rule available for qin1=%d,SL0=%.2f'%(i,qin_t1,np.round(SL_t0,2)))
            break

        #在找到合适的调度规则的情况下，进行下面步骤
        #如果根据泄流曲线调度，通过试算法迭代计算下一时刻的出库流量、水位、水库库容
        if isinstance(rule_judge_info, str):
            func_name = rule_judge_info
            ResReg.Set_discharge_func(func_name)
            var_t1,iteration,logstring = ResReg.Calc_TE_method(qin_t0,qin_t1,
                                            qout_t0,SL_t0,
                                            deltaT,epsilon)
            qout_t1,SL_t1,V_t1 = var_t1
            WriteLog(fw,'    %s   iterations %d'%(
                "routing with discharge curve",iteration
            ))
        #根据调度规则直接确定出库流量，进而确定下一时刻水库库容和水位
        else:
            qout_t1 = rule_judge_info
            WriteLog(fw,'    qout1=%.2f'%qout_t1)
            V_t1 = compute_next_V(qin_t0,qin_t1,
                                    qout_t0,qout_t1,
                                    V_t0,deltaT) #水量平衡方程
            SL_t1 = ResReg.V2SL_func(V_t1)
            
        #下一时刻出库流量、水位、水库库容
        Qout_series[i+1] = qout_t1
        SL_series[i+1]   = SL_t1
        V_series[i+1]    = V_t1
        Qout_power_series[i+1] = min(qout_t1,ResReg.POWER_DISC)

    #最后一个时刻的工业供水流量&蒸发流量&渗漏流量
    tf = time_series[-1]
    tf_obj = datetime.strptime(tf,tform)
    mon = str(tf_obj.month)
    Qext_evaporation_series[-1] = ResReg.curves_dict['EVAPORATION'][mon]/30/24/3600
    Qext_leakage_series[-1]     = ResReg.curves_dict['LEAKAGE'][mon]/30/24/3600
    if SL_series[-1] <= ResReg.curves_dict['LOWER_LIMITED_SL'][mon]:
        Qext_industry_series[-1] = 0.0
    else:
        Qext_industry_series[-1] = ResReg.curves_dict['INDUSTRY'][mon]/30/24/3600    

    #输出调洪计算结果
    header = ['time','relT','Qin','Qout','SL','V','Qpower','industry','evap','leakage']
    data = np.column_stack((
        time_series,
        np.round(relT_series,2),
        np.round(Qin_series,2),
        np.round(Qout_series,2),
        np.round(SL_series,2),
        np.round(V_series/ResReg.storage_magnitude,3),
        np.round(Qout_power_series,2),
        np.round(Qext_industry_series,4),
        np.round(Qext_evaporation_series,4),
        np.round(Qext_leakage_series,4)
    ))
    np.savetxt(output,data,delimiter=',',fmt='%s',
                header=','.join(header),comments='')
    
    CloseLog(fw)
    
    return np.array(data[:,3],dtype=float)



if __name__ == '__main__':
    print(os.getcwd())
    input_file  = sys.argv[1]
    Qout_init   = float(sys.argv[2])
    SL_init     = float(sys.argv[3])
    output_file = sys.argv[4]

    #读取水库调度定义信息
    definition_file = './reservoir_XQ.def'
    ResReg = Reservoir_Reg(definition_file)

    #入库流量
    time_series,inflow_series = Read_discdata(input_file)

    #水库调度计算
    # BTW_Routing(ResReg,time_series,inflow_series[0],
    #                    Qout_init, SL_init,
    #                    output=output_file,epsilon=0.05)
    XQ_Routing( ResReg,time_series,inflow_series[0],
                       Qout_init, SL_init,
                       output=output_file,epsilon=0.05)
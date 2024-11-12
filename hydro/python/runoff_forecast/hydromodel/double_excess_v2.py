# -*- coding: utf-8 -*-
import numpy as np
import math
from typing import Tuple,Union
from scipy.integrate import quad

def calculate_evap(lm, c, wu0, wl0, prcp, pet):
    """
    三层蒸发模型，参考：[1]曹玉涛.双超模型改进及不确定性研究[D].太原理工大学,2016.

    Three-layers evaporation model from "Watershed Hydrologic Simulation" written by Prof. RenJun Zhao.
    The book is Chinese, and its name is 《流域水文模拟》;
    The three-layers evaporation model is described in Page 76;
    The method is same with that in Page 22-23 in "Hydrologic Forecasting (5-th version)" written by Prof. Weimin Bao.
    This book's Chinese name is 《水文预报》

    Parameters
    ----------
    lm
        average soil moisture storage capacity of lower layer
    c
        coefficient of deep layer
    wu0
        initial soil moisture of upper layer; update in each time step
    wl0
        initial soil moisture of lower layer; update in each time step
    prcp
        basin mean precipitation
    pet
        potential evapotranspiration

    Returns
    -------
    tuple[np.array,np.array,np.array]
        eu/el/ed are evaporation from upper/lower/deeper layer, respectively
    """
    evapo = np.maximum(pet - prcp, 0.0)         #降雨抵消后的蒸发能力
    eu = np.where(wu0 >= evapo, evapo, wu0)
    ed = np.where((wl0 < c * lm) & (wl0 < c * (evapo - eu)), c * (evapo - eu) - wl0, 0.0)   
    el = np.where(
        wu0 >= evapo,
        0.0,
        np.where(
            wl0 >= c * lm,
            (evapo - eu) * wl0 / lm,
            np.where(wl0 >= c * (evapo - eu), c * (evapo - eu), wl0),
        ),
    )
    return eu, el, ed

def cal_Fm(t, B0, Sr, Ks, C):
    """
    计算虚构点下渗能力

    公式参考：[1]孙毅.山西省子洪水库综合自动化系统开发研究[D].太原理工大学[2023-12-16]
    注意：在[2]曹玉涛.双超模型改进及不确定性研究[D].太原理工大学,2016.中,带参数C项的都被删除了,这需要后面再确认

    Parameters
    ----------
    t
        时间
    B0
        流域平均充水度（雨前土湿)
    Sr
        充分风干土壤的宏观吸收率（即土壤的最大入渗率）
    Ks
        饱和土壤的宏观导水率（即土壤的最大导水率）
    C
        土壤孔径级配参数    
    """
    Fm = Sr * (1 - pow(B0, C + 1) ) * pow(t, 0.5) + 2 * Ks * (1 - pow(B0, 2 * C + 1) ) * t
    return Fm

def cal_eta(x, B_t, b1, β_0m):
    """
    计算供渗函数

    Parameters
    ----------
    x
        供水度, 单位时间内有效降雨量DeltaP_prime / 虚构单元可能入渗损失量delta_Fm。注意, x不能小于0, 即DeltaP_prime不能小于0
    B_t
        t时间截口的土壤饱和度,是一个中间变量
    b1
        单元体饱和时渗能指标统计分布参数,取值1.5~6。b1越大,模拟的地表径流越大
    β_0m     
        是β0的上限,一般可以取0.1

    """  
    β0 = β_0m * (1 - 2 * B_t)

    eta_1 = np.where(
        β0 >= 0,
        (1 + b1 * β0 * B_t) / (1 + b1 * B_t),
        1 / ((1 - β0) ** (b1 * B_t) * (1 + b1 * B_t))
    )
    eta =  np.where(
        x >= 1,
        eta_1, 
        np.where(
            β0 < 0,
            eta_1 - (1 - β0) / (1 + b1 * B_t) * ((1 - x) / (1 - β0)) ** (1 + b1 * B_t),
            np.where(
                x <= β0,
                x,
                eta_1 - (1 - β0) / (1 + b1 * B_t) * ((1 - x) / (1 - β0)) ** (1 + b1 * B_t)
            )
        )
    )

    return eta

def cal_t0(B0, Sr, Ks, C, Fm, ΔR, ΔF3, ΔE):
    """
    计算t0
    因为总下渗量-历时曲线中的时间t0,是由（本次降雨累计下渗量 + 土壤前期水分中参与到入渗活动中的水深- 排走或蒸发的水深）决定的。

    公式参考：[1]孙毅.山西省子洪水库综合自动化系统开发研究[D].太原理工大学[2023-12-16]

    Parameters
    ----------
    B0
        流域平均充水度（雨前土湿)
    Sr, Ks, C
        参数
    Fm
        本次降雨当前累计下渗量
    ΔR
        总壤中流径流量（三层）
    ΔF3
        第三层向下排泄量
    ΔE
        总蒸发量（三层）
    """  
    S0 = Sr * (1 - pow(B0 , (C + 1)))                       #中间变量                              
    A0 = 2 * Ks * (1 - pow(B0 , (2 * C + 1)))               #中间变量
    t0_prime = pow((pow(pow(S0 , 2) + 4 * A0 * Fm , 0.5) - S0) / (2 * A0) , 2)          
    # H0 = Sr * B0 ** (C + 1) * t0_prime ** 0.5 + 2 * Ks * B0 ** (2 * C + 1) * t0_prime           #土壤前期水分中参与到入渗活动中的水深
    # Fmm = Fm + H0 - ΔR - ΔF3 - ΔE
    # t0 = (((Sr ** 2 + 8 * Ks * Fmm) ** 0.5 - Sr) / 4 / Ks ) ** 2

    return t0_prime

def surface_runoff(Fm, t0, DeltaP_prime, B0, Sr, Ks, C, B_t, b1, β_0m, delta_t):
    """
    Parameters
    ----------
    DeltaP_prime
        扣除植物截留后的有效时段雨量
    B0
        流域平均充水度（雨前土湿)
    Sr
        充分风干土壤的宏观吸收率（即土壤的最大入渗率）
    Ks
        饱和土壤的宏观导水率（即土壤的最大导水率）
    C
        土壤孔径级配参数
    B_t
        t时间截口的土壤饱和度,是一个中间变量
    b1
        单元体饱和时渗能指标统计分布参数,取值1.5~6。b1越大,模拟的地表径流越大
    β_0m     
        是β0的上限,一般可以取0.1
    delta_t
        计算时段

    Returns
    -------
    tuple[np.array,np.array]
    时段地表径流量,下渗量 mm

    """
    delta_Fm = cal_Fm(t0 + delta_t, B0, Sr, Ks, C) - cal_Fm(t0, B0, Sr, Ks, C)
    # print("="*100)
    # print("delta_Fm:",delta_Fm)
    x = DeltaP_prime / delta_Fm
    eta = cal_eta(x, B_t, b1, β_0m)
    delta_F0 = eta * delta_Fm                   #时段入渗量
    #print("delta_F0:",delta_F0)
    delta_Rs = DeltaP_prime - delta_F0
    # Fm =  np.where(
    #     x > 1,
    #     Fm + delta_Fm, 
    #     Fm + DeltaP_prime
    # )
    #print(delta_F0)                 
    #return delta_Rs, delta_F0, Fm, delta_Fm
    return delta_Rs, delta_F0, delta_Fm

def underground_runoff(S1, W1, Sm, Wm, CC, delta_F0, Ks, b1, β_0m, eu, el, ed, δ, delta_t):
    """
    计算壤中流和地下径流
    
    "y分布曲线及壤中流的计算方法是针对某一层而言的,在实际应用时,可以是每一土层都有一种y分布曲线,
    但是这样就会造成参数过多,调试起来困难的问题.因此可以假设每一土层都有相同的 y 分布曲线,这种假设在下面的实例中被证明是可行的。"
    [1]李力,延耀兴,张海瑞.水文模拟中壤中流计算方法的研究[J].水土保持通报, 2008, 28(1)

    Parameters
    ----------
    S1
        初始重力水水深,三维数组
    W1
        初始张力水水深,三维数组
    Sm
        重力水最大水深,假设三层相同，一维数组
    Wm
        张力水最大水深,假设三层相同，一维数组
    CC 
        持蓄容量统计分配曲线形状系数,假设三层相同，一维数组
    delta_F0
        地表入渗水量(mm)
    Ks
        饱和土壤的宏观导水率（即土壤的最大导水率）  
    b1
        单元体饱和时渗能指标统计分布参数,取值1.5~6。b1越大,模拟的地表径流越大
    β_0m     
        是β0的上限,一般可以取0.1
    eu, el, ed   
        上、中、下三层土壤蒸发量(mm)
    δ
        侧排水系数
    delta_t
        时段长度,单位h
        
    """
    E = np.array([eu, el, ed])
    Ri = np.zeros([1])                  #壤中流径流量
    S2 = np.zeros([3])                  #更新后的重力水，3层
    W2 = np.zeros([3])                  #更新后的张力水，3层        
    delta_S = np.zeros([3])             #重力水变化量，3层
    delta_W = np.zeros([3])             #张力水变化量，3层
    delta_F = np.zeros([4])             #下渗量，4个值
    delta_F[0] = delta_F0
    y1 = np.zeros([3])                  #中间值
    Δy = np.zeros([3])                  #中间值
    a = 2 * Ks / (1 + b1) / (1 + β_0m) ** b1 / Sm        #a, 排泄系数
    for i in range(3):
        y1[i] = 1 - (1 - W1[i] / Wm) ** (1 / (1 + CC))
    for i in range(3):
        Δy[i] = delta_F[i] / (1 + CC) / Wm
        delta_S[i] = np.where(
            y1[i] >= 1,
            delta_F[i],
            np.where(
                (y1[i] + Δy[i]) > 1,
                W1[i] + delta_F[i] - Wm,
                W1[i] + delta_F[i] - Wm * (1 - (1 - (y1[i] + Δy[i])) ** (CC + 1))
             )
        )
        delta_W[i] = delta_F[i] - delta_S[i]
        ΔV = delta_S[i] + (S1[i] - delta_S[i] / a / delta_t) * (1 - math.e ** (-1 * a * delta_t))
        Ri = Ri + δ * ΔV                                                                #计算累积壤中流
        S2[i] = np.clip(S1[i] + delta_S[i] - ΔV, a_min=0.0001, a_max=Sm)                  #更新重力水水深,范围为0.01到Sm
        #temp = np.minimum(E[i] - (S1[i] + delta_S[i] - ΔV), 0.0)                       #蒸发先发生在重力水中，剩余部分发生在张力水中(尝试后不可行，因为蒸发量以张力水作为指标)
        W2[i] = np.clip(W1[i] + delta_W[i] - E[i], a_min=0.0001, a_max=Wm)                #蒸发只发生在张力水中，更新张力水水深,范围为0.01到Wm
        delta_F[i+1] = (1 - δ) * ΔV
    #B_t = (np.sum(W2) + np.sum(S2)) / (3 * (Sm + Wm))               #入渗前土壤充水度（流域平均充水度）（考虑3层）
    #B_t = (W2[0] + S2[0] + W2[1] + S2[1]) /  (Sm + Wm) / 2          #（考虑2层）
    B_t = (W2[0] + S2[0]) /  (Sm + Wm)                               #（考虑1层）
    #delta_Rg = delta_F[3]                                           #地下径流暂时不考虑
    # print("S:",S2)
    # print("W:",W2)
    return Ri, delta_F[3], W2, S2, B_t

def f(x,n):
    return x ** (n-1) * math.e ** (-x)

def st(t, n, k):
    '''
    计算S曲线

    Parameters
    ----------
    t
        时间
    n,k
        瞬时单位线参数

    '''
    v = t / k
    result,error = quad(f, 0, v, args = (n,) )
    st = result / math.gamma(n)             #这里的st少除了一个k吗？（答案：没有，原式的dt变为d(t/k)
    return st

def cunit_new(n, k, delta_t):
    '''
    计算时段单位线

    Parameters
    ----------
    delta_t
        时段长度
    
    '''
    uh = np.zeros(24)                                   #时段单位线长度固定为24
    for j in range(1, 24):
        t = j * delta_t
        uh[j] = st(t,n,k) - st(t-delta_t,n,k)
    return uh

def cal_Q(qt, uh, L): 
    '''
    由时段单位线计算流量过程

    Parameters
    ----------
    qt
        净雨过程
    uh
        时段单位线
    L
        输出流量的长度，最好与雨量长度相等
    
    '''                                  
    n = qt.shape[0]                                     
    m = uh.shape[0]
    Q = np.zeros(n + m - 1)
    for i in range(n):
        for j in range(m):
            Q[i + j] += qt[i] * uh[j]
    return Q[0:L]

def double_excess(   
    p_and_e,
    S1, 
    W1, 
    a0, 
    Wm,
    Sm,
    KC,
    Sr, 
    Ks, 
    C,
    b1, 
    β_0m,
    CC,
    delta_t,
    δ,
    N,
    k,
    G
):
    """
    双超产流模型

    Parameters
    ----------
    S1
        初始重力水水深,三维数组
    W1
        初始张力水水深,三维数组
    a0
        不透水面积比例
    Wm
        张力水最大水深,假设三层相同，一维数组
    Sm
        重力水最大水深,假设三层相同，一维数组        
    KC
        蒸散发能力折算系数
    Sr
        充分风干土壤的宏观吸收率（即土壤的最大入渗率）
    Ks
        饱和土壤的宏观导水率（即土壤的最大导水率）
    C
        土壤孔径级配参数
    b1
        单元体饱和时渗能指标统计分布参数,取值1.5~6。b1越大,模拟的地表径流越大
    β_0m     
        是β0的上限,一般可以取0.1
    CC 
        持蓄容量统计分配曲线形状系数,假设三层相同，一维数组
    delta_t
        计算时段,一般取1
    δ
        侧排水系数
    N
        瞬时单位线参数
    k
        瞬时单位线参数

    """
    pet = KC * p_and_e[1]                                           #蒸散发能力折算
    DeltaP_prime = np.maximum(p_and_e[0] - pet, 0.0)                #扣除蒸发后的有效降雨
    c = 0.09                                                        #深层蒸散发系数，统一假定为0.09
    L = p_and_e[0].shape[0]              
    rss_ = np.full(L, 0.0)
    ris_ = np.full(L, 0.0)
    rgs_ = np.full(L, 0.0)
    qt = np.full(L, 0.0)                                            #总出流量
    qs = np.full(L, 0.0)                                            #总径流量
    W1_Seq = []                                                     #张力水含水量序列，记录用
    S1_Seq = []                                                     #自由水含水量序列，记录用
    B_t_Seq = []                                                    #充水度序列，记录用
    t0_Seq = []                                                     #下渗时间序列，记录用
    delta_F0_Seq = []                                               #时段下渗量序列，记录用
    delta_Fm_Seq = []                                               #时段下渗能力序列，记录用
    Fm_Seq = []                                                     #累积下渗量序列，记录用
    ΔE_Seq = []                                                     #3层蒸发量序列，记录用
    #Sm = Wm                                                         #重力水容量大约为张力水容量的1倍（壤土田间持水率25%，饱和含水率50%）
    #B_t = (np.sum(W1) + np.sum(S1)) / (3 * (Sm + Wm))              #入渗前土壤充水度（流域平均充水度）（考虑3层）
    B_t = (W1[0] + S1[0]) /  (Sm + Wm)                              #（考虑1层）
    #B_t = (W1[0] + S1[0] + W1[1] + S1[1]) /  (Sm + Wm) / 2           #（考虑2层）
    B0 = B_t                                                        #使用实时雨前土湿
    # B0 = 0                                                        #认为从极度干燥时刻开始
    #Fm = np.sum(W1) + np.sum(S1)                                   #（考虑3层）
    Fm = W1[0] + S1[0]                                              #（考虑1层）
    #Fm = W1[0] + S1[0] + W1[1] + S1[1]                             #（考虑2层）
    t0 = cal_t0(B0, Sr, Ks, C, Fm, 0, 0, 0)                            

    for i in range(L):
        eu, el, ed = calculate_evap(Wm, c, W1[0], W1[1], p_and_e[0][i], pet[i])
        #(rss_[i], delta_F0, Fm, delta_Fm)= surface_runoff(Fm, t0, DeltaP_prime[i], B0, Sr, Ks, C, B_t, b1, β_0m, delta_t)
        (rss_[i], delta_F0, delta_Fm)= surface_runoff(Fm, t0, DeltaP_prime[i], B0, Sr, Ks, C, B_t, b1, β_0m, delta_t)
        (ris_[i], rgs_[i], W1, S1, B_t) = underground_runoff(S1, W1, Sm, Wm, CC, delta_F0, Ks, b1, β_0m, eu, el, ed, δ, delta_t)
        #print(W1)
        W1_Seq.append(W1)
        S1_Seq.append(S1)
        B_t_Seq.append(B_t)
        delta_F0_Seq.append(delta_F0)
        delta_Fm_Seq.append(delta_Fm)
        #Fm = np.sum(W1) + np.sum(S1)                               #（考虑3层）
        #Fm = W1[0] + S1[0] + W1[1] + S1[1]                         #（考虑2层）
        Fm = W1[0] + S1[0]                                          #（考虑1层）
        ΔE = eu + el + ed
        t0 = cal_t0(B0, Sr, Ks, C, Fm, ris_[i], rgs_[i], ΔE)
        # print("eu:",eu)
        # print("el:",el)
        # print("ed:",ed)

        t0_Seq.append(t0)
        Fm_Seq.append(Fm)
        ΔE_Seq.append(ΔE)

        rss_[i] = (1 - G) * rss_[i]
        ris_[i] = (1 - G) * ris_[i]
        rgs_[i] = rgs_[i] + G * rss_[i] + G * ris_[i]
        qt[i] = rss_[i] * (1 - a0) + ris_[i] * (1 - a0) + DeltaP_prime[i] * a0  #总径流只包含地表径流和壤中流，暂不考虑地下径流
        
    W1_Seq = np.array(W1_Seq)
    S1_Seq = np.array(S1_Seq)
    B_t_Seq = np.array(B_t_Seq)
    t0_Seq = np.array(t0_Seq)
    delta_F0_Seq = np.array(delta_F0_Seq)
    delta_Fm_Seq = np.array(delta_Fm_Seq)
    Fm_Seq = np.array(Fm_Seq)
    ΔE_Seq = np.array(ΔE_Seq)
    uh = cunit_new(N, k, delta_t)
    qs = cal_Q(qt, uh, L)
    
    return rss_, ris_, rgs_, qs, W1, S1, W1_Seq, S1_Seq, B_t_Seq, t0_Seq, delta_F0_Seq, delta_Fm_Seq, Fm_Seq, ΔE_Seq




# 根据流域面积输出流量序列和最后一个时刻的流量
def ROdepth_to_Q(qs,Area,delta_T): # mm^3/h/m^2 ~> m^3/s
    Q = (qs * Area) / 1000 / 3600 / delta_T
    return Q

# 在经过模型演算后最后一个时刻的流域状态
def Get_NewStates(S1, W1):
    states = np.tile([0.5], (1, 6))
    states[:, 0] = S1[0]
    states[:, 1] = S1[1]
    states[:, 2] = S1[2]
    states[:, 3] = W1[0]
    states[:, 4] = W1[1]
    states[:, 5] = W1[2]
    return states

def calc_double_excess(Rainfall:np.ndarray,Evaporation:np.ndarray,
                       Parameter:np.ndarray,States:np.ndarray,
                       Area:float,Hydro_Dt:float):
    
    # 降雨量和蒸发量
    p_e = np.array([Rainfall, Evaporation])

    # 模型参数
    a0 = Parameter[0]
    Wm = Parameter[1]
    Sm = Parameter[2]
    KC = Parameter[3]    
    Sr = Parameter[4]
    Ks = Parameter[5]
    C = Parameter[6]
    b1 = Parameter[7] 
    β_0m = Parameter[8]
    CC = Parameter[9]
    δ = Parameter[10]
    N = Parameter[11]
    k = Parameter[12]
    G = Parameter[13]

    # 模型状态
    S1 = States[0:3]
    W1 = States[3:6]

    delta_t = Hydro_Dt

    rss_, ris_, rgs_, qs, W1, S1, W1_Seq, S1_Seq, B_t_Seq, t0_Seq, delta_F0_Seq, delta_Fm_Seq, Fm_Seq, ΔE_Seq= double_excess(   
        p_e,
        S1, 
        W1, 
        a0, 
        Wm,
        Sm,
        KC,
        Sr, 
        Ks, 
        C,
        b1, 
        β_0m,
        CC,
        delta_t,
        δ,
        N,
        k,
        G
    )

    Q = ROdepth_to_Q(qs,Area,Hydro_Dt)
    newstate = Get_NewStates(S1,W1)

    return Q,newstate
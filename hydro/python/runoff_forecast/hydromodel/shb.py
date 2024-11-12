import math
import numpy as np

# coding=utf-8

class ModelShanbei:
    def __init__(self,value):
        self.e = math.e
        self.KC = 0.0 #蒸散发折算系数
        self.WM = 0.0 #张力水蓄水容量，一般为60~80mm
        self.FB = 0.0 #不透水面积比例
        self.f0 = 0.0 #最干旱的下渗能力，一般为1~2mm/min，即60~120mm/h,在本代码中单位为[mm/h]
        self.fc = 0.0 #稳定下渗率，一般为0.3~0.5 mm/min，即18～30mm/h，在本代码中单位为[mm/h]
        self.k = 0.0 #霍尔顿曲线方程系数，一般为0.04~0.05 /min, 即2.4~3/h，在本代码中的单位为[/h]
        self.Cs = 0.0 #地面径流消退系数，主要代表坦化作用，一般与退水流量有关	
        self.BX = 0.0 #下渗能力分布曲线抛物线指数,用于反映下渗能力在透水面积上的分布特性。B=0表示下渗能力分布均匀；B愈大，表示下渗率分布愈不均匀，B的值取决于流域的土壤结构
        self.L = 0.0 #时段数/时滞
        self.W = 0.0                #变化的土壤含水量
        self.q0 = 0.0               #增加q0赋初值（Bill)
        self.SubOut = []
        self.QS = []
        self.set_ModelPara(value)

    def set_ModelPara(self, value):
        self.KC = value[0]
        self.WM = value[1]
        self.FB = value[2]
        self.f0 = value[3]
        self.fc = value[4]
        self.k = value[5]
        self.Cs = value[6]
        self.BX = value[7]
        self.L = value[8]

    def ShanbeiMain(self, P, EM, F, W, q0, delta_T):    #P为雨量[mm]，EM为测量的蒸发量[mm]，F表示为流域面积[km2]         
        MR = {}                         #储存用的字典
        num = len(P)                    #降雨时段长度
        WCS = [None] * num              #土壤相对含水量
        fmm = [None] * num              #最大下渗能力
        f = [None] * num                #下渗率，mm/min 或 mm/h
        R = [None] * num                #表面径流量 [mm]
        L = int(self.L)                 #滞后时间
        RIMP = [None] * num             #在不透水区域的产流径流量[mm]
        MR['PR'] = [None] * num         #透水区和不透水区的总径流量 [mm]
        self.SubOut = [None] * num      #流域出口流量
        self.QS = [None] * num          #产流流量，[m3/s] 
        PE = 0.0                        #有效降雨量 (降雨量扣除蒸发量) [mm]
        E = 0.0                         #蒸发量
        U = F / 1000 / 3600 / delta_T   #F表示小流域的流域面积 [km2]，注意，如果时段长度发生改变，这里的3.6要修改！！！

        self.W = W                      #增加W赋值（Bill)
        self.q0 = q0                    #增加q0赋值（Bill)

        for i in range(num):
            E = self.KC * EM[i]         #计算蒸发量，EM为实测值，E为流域蒸发能力值
            if E < 0:
                E = 0
            if P[i] < 0:
                P[i] = 0
            PE = P[i] - E               #计算有效降雨量（降雨量扣除蒸发量）

            if PE <= 0:                 #无产流
                R[i] = 0
                RIMP[i] = 0
                self.W = self.W + PE * self.W / self.WM            #增加一层蒸发模式（Bill）
                self.QS[i] = 0
                MR['PR'][i] = R[i]
            
            if PE > 0:                  #有产流
               # f[i] = self.fc + (self.f0 - self.fc) * math.pow(self.e, self.fc * (self.f0 - self.k * self.W - self.fc))    这里公式有误(Bill)
                j = 0
                t0 = self.W / self.f0
                while j < 50:
                    W1 = self.fc * t0 + (self.f0 - self.fc) * (1 - math.pow(self.e, -1 * self.k * t0)) / self.k
                    f[i] = self.f0 - self.k * (W1 - self.fc * t0)
                    if abs(self.W - W1) < 0.05:
                        break
                    t0 = t0 + abs(W1 - self.W) / f[i]
                    j += 1

                fmm[i] = f[i] * (1 + self.BX)

                if PE >= fmm[i]: #全流域产流
                    R[i] = PE - f[i]
                    self.W = self.W + f[i]
                
                if PE < fmm[i]: #部分流域产流
                    R[i] = PE - f[i] + f[i] * math.pow(1 - PE / fmm[i], 1 + self.BX)
                    self.W = self.W + PE - R[i]
                
                RIMP[i] = PE * self.FB #不透水区的产流，有效降雨量直接产流
                R[i] = R[i] * (1 - self.FB) #透水区的产流
                MR['PR'][i] = (RIMP[i] + R[i]) #总径流量[mm]
                self.QS[i] = (RIMP[i] + R[i]) * U #产流流量，[m3/s] 
            WCS[i] = self.W / self.WM #表示时段末土壤相对含水量的大小


            if WCS[i] > 1:
                WCS[i] = 1
            self.W = self.WM * WCS[i]

        # 坡地汇流计算
        self.SubOut[-1] = self.q0                       #输入的雨量时长须大于滞后值lag(Bill)
        for i in range(L):                              #notice,when lag=0,the loop will not be excuted.需要在上一行对qs[-1,j]提前赋值
            self.SubOut[i] = self.q0                    #This should be last Q or 0. Otherwise it will cause water imbalance
        for i in range(L, num):                         #but usually the last Q is small and such error could be ignored.
            self.SubOut[i] = self.Cs * self.SubOut[i - 1] + (1 - self.Cs) * self.QS[i - L]

        MR['Q'] = self.SubOut
        MR['W'] = self.W                                #记录最后一个W值(Bill)
        return MR 
    

def calc_shb(Rainfall:np.ndarray,Evaporation:np.ndarray,
             Parameter:np.ndarray,States:np.ndarray,
             Area:float,Hydro_Dt:float):
        
    #流域平均面雨量
    P = np.maximum(Rainfall, 0.0)    
    #流域平均蒸发量
    EM = np.maximum(Evaporation, 0.0) 
    #流域初始状态
    W,q0 = States

    #计算
    shb = ModelShanbei(Parameter)
    res = shb.ShanbeiMain(P, EM, Area, W, q0, Hydro_Dt)

    #输出
    Q = res['Q']
    newstate = np.array( [ Q[-1],res['W'] ] )

    return Q,newstate

    
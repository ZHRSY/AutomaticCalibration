"""
Core code for XinAnJiang model
"""
import logging
from typing import Union
from collections import OrderedDict
import numpy as np
from numba import jit
from scipy.special import gamma


@jit(nopython=True)
def calculate_evap(lm, c, wu0, wl0, prcp, pet):
    """
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
    eu = np.where(wu0 + prcp >= pet, pet, wu0 + prcp)
    ed = np.where((wl0 < c * lm) & (wl0 < c * (pet - eu)), c * (pet - eu) - wl0, 0.0)   
    el = np.where(
        wu0 + prcp >= pet,
        0.0,
        np.where(
            wl0 >= c * lm,
            (pet - eu) * wl0 / lm,
            np.where(wl0 >= c * (pet - eu), c * (pet - eu), wl0),
        ),
    )
    return eu, el, ed


#@jit          (去掉im相关的要素，Fla)
def calculate_prcp_runoff_RR(b, wm, w0, pe, ped, nd):
    """
    Calculates the amount of runoff generated from rainfall after entering the underlying surface.

    Same in "Watershed Hydrologic Simulation" and "Hydrologic Forecasting (5-th version)"

    Parameters
    ----------
    b
        B; exponent coefficient
    wm
        average soil moisture storage capacity
    w0
        initial soil moisture
    pe
        net precipitation

    Returns
    -------
    tuple[np.array,np.array]
        r -- runoff; 
    """
    wmm = wm * (1.0 + b)     #删去(1.0-im), 原本为wm * (1.0 + b)/(1.0-im), Fla
    a = wmm * (1.0 - (1.0 - w0 / wm) ** (1.0 / (1.0 + b)))
    if np.isnan(a).any():
        raise ArithmeticError("Please check if w0>wm or b is a negative value!")
    #Newly added code for source-division are below(Bill)
    r_cal = np.full(pe.shape, 0.0)
    rd = np.full(ped.shape, 0.0)
    if pe > 0.0:
        peds = np.full(pe.shape, 0.0)
        rd = np.full(ped.shape, 0.0)
        for i in range(nd):
            a = a + ped[i]
            peds = peds + ped[i]
            rri = r_cal
            r_cal = peds - wm + w0
            if a < wmm:
                r_cal = r_cal + wm * (1.0 - a / wmm) ** (1.0 + b)
            rd[i] = r_cal - rri
    #Newly added code for source-division are above(Bill)
    r = np.maximum(r_cal, 0.0)

    return r, rd


def calculate_w_storage(
    um, lm, dm, wu0, wl0, wd0, eu, el, ed, pe, r
):
    """
    Update the soil moisture values of the three layers.

    According to the equation 2.60 in the book《水文预报》

    Parameters
    ----------
    um
        average soil moisture storage capacity of the upper layer
    lm
        average soil moisture storage capacity of the lower layer
    dm
        average soil moisture storage capacity of the deep layer
    wu0
        initial values of soil moisture in upper layer
    wl0
        initial values of soil moisture in lower layer
    wd0
        initial values of soil moisture in deep layer
    eu
        evaporation of the upper layer; it isn't used in this function
    el
        evaporation of the lower layer
    ed
        evaporation of the deep layer
    pe
        net precipitation; it is able to be negative value in this function
    r
        runoff

    Returns
    -------
    tuple[np.array,np.array,np.array]
        wu,wl,wd -- soil moisture in upper, lower and deep layer
    """
    # pe>0: the upper soil moisture was added firstly, then lower layer, and the final is deep layer
    # pe<=0: no additional water, just remove evapotranspiration,
    # but note the case: e >= p > 0
    # (1) if wu0 + p > e, then e = eu (2) else, wu must be zero
    wu = np.where(
        pe > 0.0,
        np.where(wu0 + pe - r < um, wu0 + pe - r, um),
        np.where(wu0 + pe > 0.0, wu0 + pe, 0.0),
    )
    # calculate wd before wl because it is easier to cal using where statement
    wd = np.where(
        pe > 0.0,
        np.where(wu0 + wl0 + pe - r > um + lm, wu0 + wl0 + wd0 + pe - r - um - lm, wd0),
        wd0 - ed,
    )
    # water balance (equation 2.2 in Page 13, also shown in Page 23)
    # if wu0 + p > e, then e = eu; else p must be used in upper layer,
    # so no matter what the case is, el didn't include p, neither ed
    wl = np.where(pe > 0.0, wu0 + wl0 + wd0 + pe - r - wu - wd, wl0 - el)
    # the water storage should be in reasonable range
    wu_ = np.clip(wu, a_min=0.0, a_max=um)
    wl_ = np.clip(wl, a_min=0.0, a_max=lm)
    wd_ = np.clip(wd, a_min=0.0, a_max=dm)
    return wu_, wl_, wd_

# 计算净雨
def compute_pe(p_and_e, k, lm, c, wu0=None, wl0=None, wd0=None):
    """
    Single-step runoff generation in XAJ.

    Parameters
    ----------
    p_and_e
        precipitation and potential evapotranspiration
    k
        ratio of potential evapotranspiration to reference crop evaporation
    b
        exponent parameter
    um
        average soil moisture storage capacity of the upper layer
    lm
        average soil moisture storage capacity of the lower layer
    dm
        average soil moisture storage capacity of the deep layer
    im
        impermeability coefficient
    c
        coefficient of deep layer
    wu0
        initial values of soil moisture in upper layer
    wl0
        initial values of soil moisture in lower layer
    wd0
        initial values of soil moisture in deep layer

    Returns
    -------
    tuple[tuple, tuple]
        (r, rim, e, pe), (wu, wl, wd); all variables are np.array
    """
    # make sure physical variables' value ranges are correct
    prcp = np.maximum(p_and_e[:, 0], 0.0)
    # get potential evapotranspiration
    pet = np.maximum(p_and_e[:, 1] * k, 0.0)

    # Calculate the amount of evaporation from storage
    eu, el, ed = calculate_evap(lm, c, wu0, wl0, prcp, pet)
    e = eu + el + ed

    # Calculate the runoff generated by net precipitation
    prcp_difference = prcp - e
    pe = np.maximum(prcp_difference, 0.0)  

    return pe

# 计算下渗量, Fla
def compute_fa(w,wm,fc,kf,bf,pe):
    fm  = fc*(1 + kf*(wm-w)/wm)
    fmm = fm*(1+bf)
    # print('pe:%.2f w:%.2f fm:%.2f fmm:%.2f'%(pe,w,fm,fmm))
    if pe >= fmm:
        fa = fm
    else:
        fa = fm - fm*np.power(1-pe/fmm,1+bf)
    return np.full(pe.shape,fa)

# 产流模块添加下渗入参, Fla
# 修改地表产流方式为超渗产流（透水面）和直接产流（不透水面）, Fla
# 张力水蓄满产流只发生在有下渗的区域（属于透水面区域），则去除不透水面的因素（im）, Fla
def generation(p_and_e, k, b, im, um, lm, dm, c, kf, fc, bf, wu0=None, wl0=None, wd0=None) -> tuple:
    """
    Single-step runoff generation in XAJ.

    Parameters
    ----------
    p_and_e
        precipitation and potential evapotranspiration
    k
        ratio of potential evapotranspiration to reference crop evaporation
    b
        exponent parameter
    um
        average soil moisture storage capacity of the upper layer
    lm
        average soil moisture storage capacity of the lower layer
    dm
        average soil moisture storage capacity of the deep layer
    im
        impermeability coefficient
    c
        coefficient of deep layer
    wu0
        initial values of soil moisture in upper layer
    wl0
        initial values of soil moisture in lower layer
    wd0
        initial values of soil moisture in deep layer

    Returns
    -------
    tuple[tuple, tuple]
        (r, rim, e, pe), (wu, wl, wd); all variables are np.array
    """
    # make sure physical variables' value ranges are correct
    prcp = np.maximum(p_and_e[:, 0], 0.0)    #流域平均面雨量
    # get potential evapotranspiration
    pet = np.maximum(p_and_e[:, 1] * k, 0.0) #流域平均蒸发量
    # wm
    wm = um + lm + dm
    # print('wm:%.2f um:%.2f lm:%.2f dm:%.2f'%(wm,um,lm,dm))
    if wu0 is None:
        # just an initial value
        wu0 = 0.6 * um
    if wl0 is None:
        wl0 = 0.6 * lm
    if wd0 is None:
        wd0 = 0.6 * dm
    w0_ = wu0 + wl0 + wd0

    # w0 need locate in correct range so that following calculation could be right
    # To make sure float data's calculation is correct, we'd better minus a precision (1e-5)
    w0 = np.minimum(w0_, wm - 1e-5)

    # Calculate the amount of evaporation from storage
    eu, el, ed = calculate_evap(lm, c, wu0, wl0, prcp, pet)
    e = eu + el + ed

    # Calculate the runoff generated by net precipitation
    prcp_difference = prcp - e
    pe = np.maximum(prcp_difference, 0.0)  

    #地面径流-下渗量和超渗产流计算, Fla
    rim = np.maximum(pe*im,0.0) #不透水面的产流, Fla
    # print('pe*(1.0-im) ',pe*(1.0-im))
    # print('compute_fa ',compute_fa(w0,wm,fc,kf,bf,pe*(1.0-im)))
    fa = np.minimum(
        pe*(1.0-im),
        compute_fa(w0,wm,fc,kf,bf,pe*(1.0-im))
    ) #下渗量, Fla
    rs = pe*(1.0-im) - fa #地表径流, Fla

    #下渗水量FA继续进行蓄满产流计算, Fla
    #Newly added code for source-division are below(Bill)
    if fa <= 5:
        nd = 1    
        fad = fa
    else:
        nd = int(fa/5) + 1
        fad = np.zeros(nd)
        for i in range(nd):
            fad[i] = 5
        fad[nd-1] = fa-(nd-1)*5+1e-5    
    #Newly added code for source-division are above(Bill)
    #蓄满产流不再考虑不透水面积，去除im相关要素，Fla
    rr, rrd = calculate_prcp_runoff_RR(b, wm, w0, fa, fad, nd) #rr替换r, fa替换pe, fad替换ped, 蓄满产流为地面以下径流, Fla
    # Update wu, wl, wd
    if fa > 0: #当张力水有入流时
        wu, wl, wd = calculate_w_storage(
            um, lm, dm, wu0, wl0, wd0, eu, el, ed, fa, rr  #fa替换prcp_difference,rr替换r, Fla
        )
    else: #当张力水无入流，即FA=0，PE=0，PE-E<=0，张力水蓄水量变化量为蒸发损失
        wu, wl, wd = calculate_w_storage(
            um, lm, dm, wu0, wl0, wd0, eu, el, ed, prcp_difference, rr  #fa替换prcp_difference,rr替换r, Fla
        )  
    # print("w ",wu,wl,wd)
    # print("  fa:%.4f rr:%.4f"%(fa,rr))
    return (rs, rr, rim, e, pe, fa, fad, rrd, nd), (wu, wl, wd) #增加输出量——fa,fad替换ped, Fla

def sources(pe, r, sm, ex, ki, kg, ped, rd, nd, s0=None, fr0=None) -> tuple:
    """
    Divide the runoff to different sources

    We use the initial version from the paper of the inventor of the XAJ model -- Prof. Renjun Zhao:
    "Analysis of parameters of the XinAnJiang model". Its Chinese name is <<新安江模型参数的分析>>,
    which could be found by searching in "Baidu Xueshu".
    The module's code can also be found in "Watershed Hydrologic Simulation" (WHS) Page 174.
    It is nearly same with that in "Hydrologic Forecasting" (HF) Page 148-149
    We use the period average runoff as input and the unit period is day so we don't need to difference it as books show

    We also provide code for formula from《水文预报》 the fifth version. Page 40-41 and 150-151;
    the procedures in 《工程水文学》 the third version are different we also provide.
    they are in the "sources5mm" function.

    Parameters
    ------------
    pe
        net precipitation
    r
        runoff from xaj_generation
    sm
        areal mean free water capacity of the surface layer
    ex
        exponent of the free water capacity curve
    ki
        outflow coefficients of the free water storage to interflow relationships
    kg
        outflow coefficients of the free water storage to groundwater relationships
    s0
        free water capacity of last period
    fr0
        runoff area of last period

    Return
    ------------
    tuple[tuple, tuple]
        rs -- surface runoff; ri-- interflow runoff; rg -- groundwater runoff;
        s1 -- final free water capacity;
        all variables are numpy array

    """
    # maximum free water storage capacity in a basin(流域单点最大的自由水蓄水容量)
    ms = sm * (1.0 + ex)  
    if fr0 is None:
        fr0 = 0.1
    if s0 is None:
        s0 = 0.5 * sm
    precision = 1e-5
    # For free water storage, because s is related to fr and s0 and fr0 are both values of last period,
    # we have to trans the initial value of s from last period to this one.
    # both WHS（流域水文模拟）'s sample code and HF（水文预报） use s = fr0 * s0 / fr.
    # I think they both think free water reservoir as a cubic tank. Its height is s and area of bottom rectangle is fr
    # but the problem is we will have a cubic tank with varying bottom and height, and fixed boundary (sm is fixed)
    # -> so strange !!! I think maybe 2-sources xaj is more interpretable
    # especially when r=0 then fr0=0, the free water cannot disappear immediately, so we have to use s = s0, fr=fr0
    # fr's formula could be found in Eq. 9 in "Analysis of parameters of the XinAnJiang model",
    # Here our r doesn't include rim, so there is no need to remove rim from r; this is also the method in 《水文预报》（HF）
    '''fr = np.where(r > 0.0, r / pe, fr0)
    if np.isnan(fr).any():
        raise ArithmeticError("Please check pe's data! there may be 0.0")
    ss = np.minimum(fr0 * s0 / fr, sm - precision)
    au = ms * (1.0 - (1.0 - ss / sm) ** (1.0 / (1.0 + ex)))
    if np.isnan(au).any():
        raise ValueError(
            "Error： NaN values detected. Try set clip function or check your data!!!"
        )'''
    if pe <= 0:
        rs = np.full(r.shape, 0.0)
        ri = ki * s0 * fr0
        rg = kg * s0 * fr0
        s = s0 * (1 - ki - kg)
        fr = fr0
    else:
        kss_period = (1 - (1 - (ki + kg)) ** (1 / nd)) / (1 + kg / ki)
        kg_period = kss_period * kg / ki
        rs = np.full(r.shape, 0.0)
        ri = np.full(r.shape, 0.0)
        rg = np.full(r.shape, 0.0)
        td = np.full(rd.shape, 0.0)
        fr = fr0
        s = s0
        for i in range(nd):
            td = rd[i]                  #去掉im相关要素 - im * ped[i], Fla
            xx = fr                     #前一时段的产流面积比
            fr = td / ped[i]            #当前时段产流面积比
            s = xx * s / fr
            if s >= sm:
                rr = fr * (ped[i] + s - sm)
            else:
                au = ms * (1.0 - (1.0 - s / sm) ** (1.0 / (1.0 + ex)))
                ff = au + ped[i]
                if ff < ms:
                    ff = (1-(ped[i]+au)/ms)** (1.0 + ex)
                    rr = (ped[i] - sm + s + ff * sm) * fr
                else:
                    rr = fr * (ped[i] + s - sm)
            rs = rr + rs
            s = ped[i] -rr/fr + s
            ri = kss_period * s * fr + ri
            rg = kg_period * s * fr + rg
            s = s * (1 - kss_period-  kg_period)

    s1 = s                   #np.clip(s, a_min=np.full(s.shape, 0.0), a_max=sm)
    return (rs, ri, rg), (s1, fr) 


@jit(nopython=True)
def linear_reservoir(x, weight, last_y=None) -> np.array:
    """
    Linear reservoir's release function

    Parameters
    ----------
    x
        the input to the linear reservoir
    weight
        the coefficient of linear reservoir
    last_y
        the output of last period

    Returns
    -------
    np.array
        one-step forward result
    """
    weight1 = 1 - weight
    if last_y is None:
        last_y = np.full(weight.shape, 0.001)
    y = weight * last_y + weight1 * x
    return y


def uh_conv(x, uh_from_gamma):
    """
    Function for 1d-convolution calculation

    Parameters
    ----------
    x
        x is a sequence-first variable; the dim of x is [seq, batch, feature=1];
        feature must be 1
    uh_from_gamma
        unit hydrograph from uh_gamma; the dim: [len_uh, batch, feature=1];
        feature must be 1

    Returns
    -------
    np.array
        convolution
    """
    outputs = np.full(x.shape, 0.0)
    time_length, batch_size, feature_size = x.shape
    if feature_size > 1:
        logging.error("We only support one-dim convolution now!!!")
    for i in range(batch_size):
        uh = uh_from_gamma[:, i, 0]
        inputs = x[:, i, 0]
        outputs[:, i, 0] = np.convolve(inputs, uh)[:time_length]
    return outputs


def uh_gamma(a, theta, len_uh=15):
    """
    A simple two-parameter Gamma distribution as a unit-hydrograph to route instantaneous runoff from a hydrologic model
    The method comes from mizuRoute -- http://www.geosci-model-dev.net/9/2223/2016/

    Parameters
    ----------
    a
        shape parameter
    theta
        timescale parameter
    len_uh
        the time length of the unit hydrograph
    Returns
    -------
    torch.Tensor
        the unit hydrograph, dim: [seq, batch, feature]
    """
    # dims of a: time_seq (same all time steps), batch, feature=1
    m = a.shape
    if len_uh > m[0]:
        raise RuntimeError(
            "length of unit hydrograph should be smaller than the whole length of input"
        )
    # aa > 0, here we set minimum 0.1 (min of a is 0, set when calling this func); First dimension of a is repeat
    aa = np.maximum(0.0, a[0:len_uh, :, :]) + 0.1
    # theta > 0, here set minimum 0.5
    theta = np.maximum(0.0, theta[0:len_uh, :, :]) + 0.5
    # len_f, batch, feature
    t = np.expand_dims(
        np.swapaxes(np.tile(np.arange(0.5, len_uh * 1.0), (m[1], 1)), 0, 1), axis=-1
    )
    denominator = gamma(aa) * (theta**aa)
    # [len_f, m[1], m[2]]
    w = 1 / denominator * (t ** (aa - 1)) * (np.exp(-t / theta))
    w = w / w.sum(0)  # scale to 1 for each UH
    return w


def xaj_vmrm(
    p_and_e,
    params: Union[np.array, list],
    states,
    return_state=False,
    kernel_size=15,
    warmup_length=30,
    route_method="CSL",
    source_type="sources",
    source_book="ShuiWenYuBao",
) -> Union[tuple, np.array]:
    """
    run XAJ model

    Parameters
    ----------
    p_and_e
        prcp and pet; sequence-first (time is the first dim) 3-d np array: [time, basin, feature=2]
    params
        parameters of XAJ model for basin(s);
        2-dim variable -- [basin, parameter]:
        the parameters are B IM UM LM DM C SM EX KI KG A THETA CI CG (notice the sequence)
    return_state
        if True, return state values, mainly for warmup periods
    kernel_size
        the length of unit hydrograph
    warmup_length
        hydro models need a warm-up period to get good initial state values
    route_method
        now we provide two ways: "CSL" (recession constant + lag time) and "MZ" (method from mizuRoute)
    source_type
        default is "sources" and it will call "sources" function; the other is "sources5mm",
        and we will divide the runoff to some <5mm pieces according to the books in this case
    source_book
        When source_type is "sources5mm" there are two implementions for dividing sources,
        as the methods in "ShuiWenYuBao" and "GongChengShuiWenXue"" are different.
        Hence, both are provided, and the default is the former.

    Returns
    -------
    Union[np.array, tuple]
        streamflow or (streamflow, states)
    """
    # params
    if route_method == "CSL":
        param_ranges = OrderedDict(
            {
                "K": [0.5, 2.0],
                "B": [0.1, 0.5],
                "IM": [0.0, 0.2],
                "UM": [0.0, 30.0],
                "LM": [50.0, 90.0],
                "DM": [20.0, 120.0],
                "C": [0.0, 0.4],
                "SM": [1, 100.0],
                "EX": [1.0, 1.5],
                "KI": [0.0, 0.7],
                "KG": [0.0, 0.7],
                "CS": [0.0, 1.0],
                "L": [0.0, 10.0],  # the unit is one time_interval_hours
                "CI": [0.0, 0.999],
                "CG": [0.1, 0.9999],

                # 垂向混合参数, Fla
                "KF": [0.0, 2.0], #渗透系数, Fla
                "FC": [0.0, 40.0], #稳定下渗率, Fla
                "BF": [0.2, 2.0], #下渗分布曲线指数, Fla
                } )
    elif route_method == "MZ":
        param_ranges = OrderedDict(
            {
                "K": [0.5, 2.0],
                "B": [0.1, 0.4],
                "IM": [0.01, 0.1],
                "UM": [0.0, 20.0],
                "LM": [60.0, 90.0],
                "DM": [60.0, 120.0],
                "C": [0.0, 0.2],
                "SM": [1, 100.0],
                "EX": [1.0, 1.5],
                "KI": [0.0, 0.7],
                "KG": [0.0, 0.7],
                "A": [0.0, 2.9],
                "THETA": [0.0, 6.5],
                "CI": [0.0, 0.9],
                "CG": [0.98, 0.998],
            }
        )
    else:
        raise NotImplementedError(
            "We don't provide this route method now! Please use 'CS' or 'MZ'!"
        )
    xaj_params = [
        (value[1] - value[0]) * params[:, i]+value[0] 
        for i, (key, value) in enumerate(param_ranges.items())]                   
    k = xaj_params[0]
    b = xaj_params[1]
    im = xaj_params[2]
    um = xaj_params[3]
    lm = xaj_params[4]
    dm = xaj_params[5]
    c = xaj_params[6]
    sm = xaj_params[7]
    ex = xaj_params[8]
    ki = xaj_params[9]
    kg = xaj_params[10]
    # ki+kg should be smaller than 1; if not, we scale them
    ki = np.where(ki + kg < 1.0, ki, 1 / (ki + kg) * ki)
    kg = np.where(ki + kg < 1.0, kg, 1 / (ki + kg) * kg)
    if route_method == "CSL":
        cs = xaj_params[11]
        l = xaj_params[12]
    elif route_method == "MZ":
        # we will use routing method from mizuRoute -- http://www.geosci-model-dev.net/9/2223/2016/
        a = xaj_params[11]
        theta = xaj_params[12]
    else:
        raise NotImplementedError(
            "We don't provide this route method now! Please use 'CS' or 'MZ'!"
        )
    ci = xaj_params[13]
    cg = xaj_params[14]

    #垂向混合参数, Fla
    kf = xaj_params[15] #渗透系数, Fla
    fc = xaj_params[16] #稳定下渗率, Fla
    bf = xaj_params[17] #下渗分布曲线指数, Fla

    # initialize state values
    if warmup_length > 0:
        p_and_e_warmup = p_and_e[0:warmup_length, :, :]
        _, *w0, s0, fr0, qi0, qg0 = xaj_vmrm(
              p_and_e_warmup,
              params,
              return_state=True,
              kernel_size=kernel_size,
              warmup_length=0,
          )
    else:
        w0 = (states[:, 0], 1.0 * states[:, 1], 1.0 * states[:, 2])#Notice,changed from 0.5 to 0.9 here.(changed by Bill)
        s0 =  np.full(sm.shape, states[:, 3])
        fr0 = np.full(ex.shape, states[:, 4])
        qi0 = np.full(ci.shape, states[:, 5])
        qg0 = np.full(cg.shape, states[:, 6])

        #垂向混合模型初始状态——初始下渗量可通过公式计算, Fla

    # state_variables
    inputs = p_and_e[warmup_length:, :, :]     #p_and_e：sequence-first (time is the first dim) 3-d np array: [time, basin, feature=2]
    runoff_rs_= np.full(inputs.shape[:2], 0.0) #地表径流，为超渗产流, Fla
    runoff_rr_= np.full(inputs.shape[:2], 0.0) #地面以下径流，为下渗量带来的张力水蓄满产流, Fla
    runoff_rrs_= np.full(inputs.shape[:2], 0.0) #地面以下径流带来的自由水蓄满产流, Fla
    e_= np.full(inputs.shape[:2], 0.0)
    s_= np.full(inputs.shape[:2], 0.0)
    fr_= np.full(inputs.shape[:2], 0.0)
    runoff_ims_ = np.full(inputs.shape[:2], 0.0)
    rss_ = np.full(inputs.shape[:2], 0.0)
    ris_ = np.full(inputs.shape[:2], 0.0)
    rgs_ = np.full(inputs.shape[:2], 0.0)
    for i in range(inputs.shape[0]):
        # print('time:%d p_and_e'%i,inputs[i, :, :])
        if i == 0:
            (rs, rr, rim, e, pe, fa, fad, rrd, nd), w = generation(  #增加输出量——地面径流rs、下渗量fa, 修改张力水蓄满产流r为地面以下径流rr, Fla
                inputs[i, :, :], k, b, im, um, lm, dm, c, kf, fc, bf, *w0
            )
            runoff_rs_[i, :]= rs #储存地面径流, Fla
            runoff_rr_[i, :]= rr #储存地面以下径流, Fla
            e_[i, :]= e
            
            #对地面以下径流RR进行分水源操作, Fla
            if source_type == "sources":
                (rrs, ri, rg), (s, fr) = sources(fa, rr, sm, ex, ki, kg, fad, rrd, nd, s0, fr0) #去除输出量——rs, 修改净雨pe为下渗量fa,ped为fad,张力水蓄满产流r为地面以下径流rr, Fla
            else:
                raise NotImplementedError("No such divide-sources method")
            runoff_rrs_[i, :]= rrs #储存地面以下径流带来的自由水蓄满产流, Fla

            s_[i, :]= s
            fr_[i, :]= fr     
        else:
            (rs, rr, rim, e, pe, fa, fad, rrd, nd), w = generation(
                inputs[i, :, :], k, b, im, um, lm, dm, c, kf, fc, bf, *w          
            )
            runoff_rs_[i, :]= rs #储存地面径流, Fla
            runoff_rr_[i, :]= rr #储存地面以下径流, Fla
            e_[i, :]= e

            #对地面以下径流RR进行分水源操作, Fla
            if source_type == "sources":
                (rrs, ri, rg), (s, fr) = sources(fa, rr, sm, ex, ki, kg, fad, rrd, nd, s, fr)
            else:
                raise NotImplementedError("No such divide-sources method")
            runoff_rrs_[i, :]= rrs #储存地面以下径流带来的自由水蓄满产流, Fla
            s_[i, :]= s
            fr_[i, :]= fr

        # print('e_:%.4f; rs:%.4f; rr:%.4f; rim:%.4f; pe:%.4f'%(e,rs,rr,rim,pe))
        # print('w:',w)
        # print("s:%.4f; fr:%.4f"%(s,fr))
        # print("rs:%.4f; ri:%.4f; rg:%.4f"%(rs,ri,rg))
        # print('*'*40)
        # impevious part is pe * im
        runoff_ims_[i, :] = rim
        # so for non-imprvious part, the result should be corrected
        rss_[i, :] = rs+rrs      #超渗产流+张力水蓄满产流后经自由水蓄满产流, Fla
        ris_[i, :] = ri          
        rgs_[i, :] = rg          
    # seq, batch, feature
    runoff_im = np.expand_dims(runoff_ims_, axis=2)
    rss = np.expand_dims(rss_, axis=2)

    # print("="*50)
    # print("qi0:%.4f; qg0:%.4f"%(qi0,qg0))

    qs = np.full(inputs.shape[:2], 0.0)
    if route_method == "CSL":
        qt = np.full(inputs.shape[:2], 0.0)
        for i in range(inputs.shape[0]):
            if i == 0:
                qi = linear_reservoir(ris_[i], ci, qi0)
                qg = linear_reservoir(rgs_[i], cg, qg0)
            else:
                qi = linear_reservoir(ris_[i], ci, qi)
                qg = linear_reservoir(rgs_[i], cg, qg)
            qs_ = rss_[i] + runoff_ims_[i]
            qt[i, :] = qs_ + qi + qg
            # print("qs:%.4f; qi:%.4f; qg:%.4f; qt:%.4f"%(qs_,qi,qg,qt[i, :]))
        for j in range(len(l)):
            lag = int(l[j])
            qs[0,j] = states[:, 7]
            for i in range(lag):                    #notice,when lag=0,the loop will not be excuted.需要在上一行对qs[0,j]提前赋值
                qs[i, j] = states[:, 7]              #This should be last Q or 0. Otherwise it will caused water imbalance(Bill)
            for i in range(lag, inputs.shape[0]):     #but usually the last Q is small and such error could be ignored.
                qs[i, j] = cs[j] * qs[i - 1, j] + (1 - cs[j]) * qt[i - lag, j]
    elif route_method == "MZ":
        rout_a = a.repeat(rss.shape[0]).reshape(rss.shape)
        rout_b = theta.repeat(rss.shape[0]).reshape(rss.shape)
        conv_uh = uh_gamma(rout_a, rout_b, kernel_size)
        qs_ = uh_conv(runoff_im + rss, conv_uh)
        for i in range(inputs.shape[0]):
            if i == 0:
                qi = linear_reservoir(ris_[i], ci, qi0)
                qg = linear_reservoir(rgs_[i], cg, qg0)
            else:
                qi = linear_reservoir(ris_[i], ci, qi)
                qg = linear_reservoir(rgs_[i], cg, qg)
            qs[i, :] = qs_[i, :, 0] + qi + qg
    else:
        raise NotImplementedError(
            "We don't provide this route method now! Please use 'CS' or 'MZ'!"
        )

    # seq, batch, feature
    q_sim = np.expand_dims(qs, axis=2)
    if return_state:
        return q_sim, *w, s, fr, qi, qg,runoff_rs_[:],runoff_rr_[:],runoff_ims_[:],e_[:],s_[:],fr_[:],rss_,ris_,rgs_
        #0-q_sim; 1-um; 2-lm; 3-dm; 4-s; 5-fr; 6-qi; 7-qg; 8-runoff_rs_[:]; 9-runoff_rr_[:]; 10-runoff_ims_[:];
        #11-e_[:]; 12-s_[:]; 13-fr_[:]; 14-rss_; 15-ris_; 16-rgs_;
    return q_sim


# 以下Fla
# 根据流域面积输出流量序列和最后一个时刻的流量
# delta_T: 单位时间，单位h
def depth_to_Q(res,Area,delta_T): # mm^3/Dt/m^2 ~> m^3/s
    res_val = res[0]
    Q       = (res_val[: , :, 0] * Area) / 1000 / 3600 / delta_T
    return Q[:,0]

# 在经过新安江模型演算后最后一个时刻的土壤状态和流量，垂向混合产流无额外状态参数
# delta_T: 单位时间，单位h
# res_val, wu_val, wl_val, wd_val, s_val, fr_val, qi_val, qg_val, runoff_val, runoff_ims_val, e_, s_, fr_, rss_, ris_, rgs_
def Get_NewStates(res,delta_T):
    res_val, wu_val, wl_val, wd_val, s_val, fr_val, qi_val, qg_val = res[0:8]
    states = np.tile([0.5], (1, 8))
    states[:, 0] = wu_val  # wu0
    states[:, 1] = wl_val  # wl0
    states[:, 2] = wd_val  # wd0
    states[:, 3] = s_val  # s0
    states[:, 4] = fr_val  # fr0
    states[:, 5] = qi_val/delta_T  # qi0
    states[:, 6] = qg_val/delta_T  # qg0
    states[:, 7] = res_val[-1, 0, 0]/delta_T  # qt0, units: mm^3/h/m^2
    return states

def calc_vmrm(Rainfall:np.ndarray,Evaporation:np.ndarray,
             Parameter:np.ndarray,States:np.ndarray,
             Area:float,Hydro_Dt:float) -> Union[np.ndarray,np.ndarray]:

    # 降雨量和蒸发量
    p_e = np.array([Rainfall, Evaporation]).T
    p_e = np.expand_dims(p_e, axis=1)
    # 模型参数
    params = np.array(Parameter)
    params = np.expand_dims(params, axis=0)
    # 上一时段土壤状态和流量
    states_data = np.array(States)
    states_data = np.expand_dims(states_data, axis=0)
    res = xaj_vmrm(p_e,
            params,
            states_data,
            return_state=True,
            warmup_length=0)
    Q = depth_to_Q(res,Area,Hydro_Dt)
    newstate = Get_NewStates(res,Hydro_Dt)

    return Q,newstate
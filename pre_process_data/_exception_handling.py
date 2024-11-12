
import numpy as np
import pandas as pd

def oulier_detection(flow_data : list) -> list:


   '''
   flow_data 中的 None 替换为nan（即 NumPy 中的“缺失值”）。。
   将数组中为 0 的元素替换为 np.nan。
   插值处理:
   线性插值，填补 nan，并将结果转回 NumPy 数组。
   窗口滑动异常检测:
   定义窗口大小 window_size = 3。
   从第 window_size 个元素开始，进行滑动窗口操作：
   计算窗口内的均值 mean。
   取窗口内的4个数据：a, b, c, d。
   如果第一个元素 a 与均值的差异比窗口内其他元素更大，认为 a 可能是异常值。
   进一步计算一个子窗口的均值 mean1，如果 a 与这个均值的差异超过 40%，则将 a 设置为 0。
   将数组中为 0 的值再次替换为 np.nan，并进行插值操作。
   将两端的 NaN 值替换为 0:
   将结果中所有 nan 替换为 0，确保最终结果不包含缺失值。

   '''
   flow_data = [x if x is not None else np.nan for x in flow_data]
   flow_array = np.array(flow_data)
   optimized_flow = flow_array.copy()
   optimized_flow = np.where(optimized_flow == 0, np.nan, optimized_flow)
   
   optimized_flow = pd.Series(optimized_flow).interpolate().to_numpy()
   window_size = 3
   for i in range(window_size, len(optimized_flow) - 1):
      window = optimized_flow[i - window_size:i]
      mean = np.mean(window)
      a = optimized_flow[i - window_size]
      b, c, d = optimized_flow[i - window_size + 1:i - window_size + 4]

      if abs(a - mean) > max(abs(b - mean), abs(c - mean), abs(d - mean)):
         window1 = optimized_flow[i - window_size + 1:i]
         mean1 = np.mean(window1)
         if abs(a - mean1) > 0.4 * mean1:
               optimized_flow[i - window_size] = 0
   optimized_flow = np.where(optimized_flow == 0, np.nan, optimized_flow)
   optimized_flow = pd.Series(optimized_flow).interpolate().to_numpy()
   #两头的nan等于0
   optimized_flow = np.where(np.isnan(optimized_flow), 0, optimized_flow)




   return optimized_flow.tolist()

# flow_data = [0, 10, 15, 50, 14, 13, 12, 11, 9, 0]
# optimized_data = oulier_detection(flow_data)
# print(optimized_data)
import numpy as np

file = "/home/zhr/Project/Automatic_calibration_parameters/hydro/results/20240927-0h/input/PRCP_voro.csv"
data = np.loadtxt(file,delimiter=',',dtype=str)

new_line = np.zeros(60)
new_line[5:10] = [1,15,30,6,2]
new_line[15:20] = 0
new_line[25:30] = 0
new_line[35:40] = 0
new_line[45:50] = [1,5,10,6,2]
data[1:,1] = new_line

np.savetxt("/home/zhr/Project/Automatic_calibration_parameters/hydro/results/20241012_auto/input/PRCP_voro.csv",
           data,delimiter=',',fmt="%s")



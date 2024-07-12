# Create script to run numpy IEKF filter - for testing


from IEKF.iekf_filter import IEKF
import time
import numpy as np
import matplotlib.pyplot as plt
from Utils.utils import read_raw_oxts_data
from Utils.utils_plot import plot_route, plot_2D_error


oxts_file_loc = '../data/2011_09_26_drive_0009_extract/'


t, ang_gt, p_gt, v_gt, u = read_raw_oxts_data(oxts_file_loc)


# initialize the IEKF
iekf = IEKF()




N = None
start_time = time.time()
measurements_covs = np.array([0.01,0.01])
Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = iekf.run(t, u, measurements_covs, v_gt, p_gt, N,ang_gt[0])
diff_time = time.time() - start_time
print("Execution time: {:.2f} s (sequence time: {:.2f} s)".format(diff_time,t[-1] - t[0]))



plot_route(p_gt, p)
plot_2D_error(t, p_gt, p)
plt.show()
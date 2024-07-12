#####################################################
# Main script to run numpy EKF filter. 
# User should change the variable 'oxts_file_loc' to 
# their specific data file/location before running.
#####################################################


from EKF.ekf_filter import EKF
import time
import numpy as np
import matplotlib.pyplot as plt
from Utils.utils import read_raw_oxts_data
from Utils.utils_plot import plot_route, plot_2D_error


# Set file location of oxts data folder
oxts_file_loc = '../data/2011_09_30_drive_0028_extract/'


# get data from file
t, ang_gt, p_gt, v_gt, u = read_raw_oxts_data(oxts_file_loc)


# initialize the EKF
ekf = EKF()

# set length of run by number of iterations
# If N = None, filter will run through number of iterations equal
# to the number of data points in p_gt
N = None

# set values to use in measurement matrix H
measurements_covs = np.array([0.01,0.01])

# set start time based upon current system time
# for calculating how long the filter takes to run
start_time = time.time()

# set lever arm between vehicle center & IMU
t_c_i = [-.5, 0.22, 0.88]

# set rotation matrix between vehicle center and IMU
Rot_c_i = np.array([[1,0,0],[0,1,0],[0,0,1]])

# Run the filter based upon input IMU data, covariance data, and initial velocity & angle values
Rot, v, p, b_omega, b_acc = ekf.run(N, t, u, measurements_covs, v_gt[0], ang_gt[0], Rot_c_i, t_c_i)

# calculate total time that filter ran
diff_time = time.time() - start_time
print("Execution time: {:.2f} s (sequence time: {:.2f} s)".format(diff_time,t[-1] - t[0]))

# Create plots to show off filter accuracy
plot_route(p_gt, p)
plot_2D_error(t, p_gt, p)
plt.show()




import numpy as np
from Utils.utils import from_rpy, rot_angles_to_matrix_XYZ, skew, normalize_rot, rotation_integral


class EKF:
    """Create class to hold filter functions, variables, and constants"""
    Id2 = np.eye(2)
    Id3 = np.eye(3)
    Id6 = np.eye(6)
    IdP = np.eye(15)

    def __init__(self, parameter_class=None):
        """Initialize class, set all parameters to None types."""

        # variables to initialize with `filter_parameters`
        self.g = None
        self.cov_omega = None
        self.cov_acc = None
        self.cov_b_omega = None
        self.cov_b_acc = None
        self.cov_b_omega0 = None
        self.cov_b_acc0 = None
        self.cov_Rot0 = None
        self.cov_v0 = None
        self.Q = None
        self.Q_dim = None
        self.n_normalize_rot = None
        self.P_dim = None
        self.t_c_i = None
        self.Rot_c_i = None

        # set the parameters
        if parameter_class is None:
            filter_parameters = EKF.Parameters()
        else:
            filter_parameters = parameter_class()
        self.filter_parameters = filter_parameters
        self.set_param_attr()


    class Parameters:
        """Set constants to use within the filter."""

        g = np.array([0, 0, -9.80665])
        """gravity vector"""

        P_dim = 15
        """covariance dimension"""

        Q_dim = 12
        """process noise covariance dimension"""

        # Process noise covariance
        cov_omega = 2e-4
        """gyro covariance"""

        cov_acc = 1e-3
        """accelerometer covariance"""

        cov_b_omega = 1e-8
        """gyro bias covariance"""

        cov_b_acc = 1e-6
        """accelerometer bias covariance"""

        cov_Rot0 = 1e-6
        """initial pitch and roll covariance"""

        cov_b_omega0 = 1e-8
        """initial gyro bias covariance"""

        cov_b_acc0 = 1e-3
        """initial accelerometer bias covariance"""

        cov_v0 = 1e-1
        """initial velocity covariance"""

        # numerical parameters
        n_normalize_rot = 100
        """timestamp before normalizing orientation"""


        def __init__(self, **kwargs):
            self.set(**kwargs)


        def set(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)



    def set_param_attr(self):
        """set parameters based upon parameters stored within parameters class. Also initialize Q matrix."""
        
        # get a list of attribute only
        attr_list = [a for a in dir(self.filter_parameters) if not a.startswith('__')
                     and not callable(getattr(self.filter_parameters, a))]
        for attr in attr_list:
            setattr(self, attr, getattr(self.filter_parameters, attr))

        self.Q = np.diag([self.cov_omega, self.cov_omega, self. cov_omega,
                           self.cov_acc, self.cov_acc, self.cov_acc,
                           self.cov_b_omega, self.cov_b_omega, self.cov_b_omega,
                           self.cov_b_acc, self.cov_b_acc, self.cov_b_acc])
                           


    def run(self, N, t, u, measurements_covs, v0, ang0, Rot_c_i, t_c_i):
        """Main function to run the filter. Will iterate through each IMU data point and
        run the propagate & update steps to calculate the new position/velocity/attitude over time"""
        
        dt = t[1:] - t[:-1]  # (s)
        if N is None:
            N = u.shape[0]

        # Initialize the arrays/matrices used for storing states. Also set initial values to fed parameters.
        Rot, v, p, b_omega, b_acc, P = self.init_run(v0, ang0, Rot_c_i, t_c_i, N)

        # Run through each iteration
        for i in range(1, N):

            # First, run the propagate step:
            Rot[i], v[i], p[i], b_omega[i], b_acc[i], P = self.propagate(Rot[i-1], v[i-1], p[i-1], b_omega[i-1], b_acc[i-1], P, u[i], dt[i-1])

            # Then run the update step
            Rot[i], v[i], p[i], b_omega[i], b_acc[i], P = self.update(Rot[i], v[i], p[i], b_omega[i], b_acc[i], P, u[i], measurements_covs)
            
            # correct numerical error every second
            if i % self.n_normalize_rot == 0:
                Rot[i] = normalize_rot(Rot[i])

        return Rot, v, p, b_omega, b_acc



    def init_run(self, v0, ang0, Rot_c_i, t_c_i, N):
        """Initialize arrays to hold states. 
        Also set initial velocity and attitude values based upon inputs."""

        Rot, v, p, b_omega, b_acc = self.init_saved_state(N)

        # set rotation matrix based upon input Euler angles
        Rot[0] = from_rpy(ang0[0], ang0[1], ang0[2])

        # set initial velocity
        v[0] = v0

        # set rotation between vehicle and IMU
        self.Rot_c_i = Rot_c_i

        # set lever arm between vehicle and IMU
        self.t_c_i = t_c_i

        # Initialize covariance matrix, P
        P = self.init_covariance()

        return Rot, v, p, b_omega, b_acc, P



    def init_covariance(self):
        """Initialize covariance array P"""

        P = np.zeros((self.P_dim, self.P_dim))
        P[:2, :2] = self.cov_Rot0*self.Id2  
        P[3:5, 3:5] = self.cov_v0*self.Id2
        P[9:12, 9:12] = self.cov_b_omega0*self.Id3
        P[12:15, 12:15] = self.cov_b_acc0*self.Id3
        return P



    def init_saved_state(self, N):
        """Create numpy arrays (of length N) to store state variables and calculated values"""

        Rot = np.zeros((N, 3, 3))
        v = np.zeros((N, 3))
        p = np.zeros((N, 3))
        b_omega = np.zeros((N, 3))
        b_acc = np.zeros((N, 3))
        return Rot, v, p, b_omega, b_acc



    def propagate(self, Rot_prev, v_prev, p_prev, b_omega_prev, b_acc_prev, P_prev, u, dt):
        """Function runs the propogate step of the filter.
        States are propgated forward based upon previous values."""

        # Propagate acceleration based upon input IMU values, subtract the estimated accelerator
        # biases, then add gravity term (IMUs experience opposite force due to gravity)
        acc = Rot_prev.dot(u[3:6] - b_acc_prev) + self.g

        # Propagate velocity based upon new acceleleration
        v = v_prev + acc * dt

        # Propagate position based upon new acceleration
        p = p_prev + v_prev*dt + 1/2 * acc * dt**2

        # Calculate corrected gyro angle based upon input IMU value and subtract estimated gyro bias
        omega = u[:3] - b_omega_prev

        # Calculate updated rotation matrix based upon corrected gyro rate
        Rot = Rot_prev.dot(rotation_integral(omega * dt))

        # Don't update bias terms 
        b_omega = b_omega_prev
        b_acc = b_acc_prev
        
        # Propagate forward covariance
        P = self.propagate_cov(P_prev, Rot_prev, v_prev, p_prev, dt)

        return Rot, v, p, b_omega, b_acc, P



    def propagate_cov(self, P_prev, Rot_prev, v_prev, p_prev, dt):
        """Propogate the covariance array P based upon previous values."""

        F = np.zeros((self.P_dim, self.P_dim))
        G = np.zeros((self.P_dim, self.Q_dim))
        v_skew_rot = skew(v_prev).dot(Rot_prev)
        p_skew_rot = skew(p_prev).dot(Rot_prev)

        F[3:6, :3] = skew(self.g)
        F[6:9, 3:6] = self.Id3
        G[3:6, 3:6] = Rot_prev
        F[3:6, 12:15] = -Rot_prev
        G[:3, :3] = Rot_prev
        G[3:6, :3] = v_skew_rot
        G[6:9, :3] = p_skew_rot
        F[:3, 9:12] = -Rot_prev
        F[3:6, 9:12] = -v_skew_rot
        F[6:9, 9:12] = -p_skew_rot
        G[9:15, 6:12] = self.Id6

        F = F * dt
        G = G * dt
        F_square = F.dot(F)
        F_cube = F_square.dot(F)
        Phi = self.IdP + F + 1/2*F_square + 1/6*F_cube
        P = Phi.dot(P_prev + G.dot(self.Q).dot(G.T)).dot(Phi.T)

        return P



    def update(self, Rot, v, p, b_omega, b_acc, P, u, measurement_cov):
        """Run the update step of the filter. This will use the propagated states and
        'fictitious' measurements to calculate the innovation, then utilize measurement and 
        process noise estimates to update the states."""

        # calculate rotation matrix of body frame
        Rot_body = Rot.dot(self.Rot_c_i)

        # calculate velocity in imu frame
        v_imu = Rot.T.dot(v)

        # calculate velocity in body frame
        v_body = self.Rot_c_i.T.dot(v_imu)

        # calculate velocity in body frame in the vehicle axis
        v_body += skew(self.t_c_i).dot(u[:3] - b_omega)

        # calculate corrected gyro bias matrix
        Omega = skew(u[:3] - b_omega)

        # Calculate Jacobian matrix H
        # H is w.r.t. car frame
        H_v_imu = self.Rot_c_i.T.dot(skew(v_imu))
        H_t_c_i = -skew(self.t_c_i)

        H = np.zeros((2, self.P_dim))
        H[:, 3:6] = Rot_body.T[1:]
        H[:, 9:12] = H_t_c_i[1:]

        # innovation is calculated as velocity in vehicle lateral and up/down directions
        # this is a 'fictitious' measurement since it's not directly measured, but we know
        # from dynamics that the instaneous velocity of a vehicle in these directions should be 
        # zero, or at least near zero.
        # NOTE that these are calculated to be able to run the full filter. A Kalman filter is
        # dependent on input measurements, and so the filter needed some kind of data (measureable or not)
        # to be able to update the filter.
        r = - v_body[1:]

        # set measurement noise matrix
        R = np.diag(measurement_cov)

        # calculate Kalman Gain
        S = H.dot(P).dot(H.T) + R
        K = (np.linalg.solve(S, P.dot(H.T).T)).T

        # calculate change in states, and use to calculate new states
        dx = K.dot(r)
        dR = rot_angles_to_matrix_XYZ(dx[0],dx[1],dx[2])

        # calculate new rotation matrix
        Rot_up = dR.dot(Rot)

        # calculate new velocity and position
        dv = dx[3:6]
        dp = dx[6:9]
        v_up = dR.dot(v) + dv
        p_up = dR.dot(p) + dp

        # calculate new bias terms
        b_omega_up = b_omega + dx[9:12]
        b_acc_up = b_acc + dx[12:15]

        # Update covariance matrix
        I_KH = EKF.IdP - K.dot(H)
        P_up = I_KH.dot(P).dot(I_KH.T) + K.dot(R).dot(K.T)
        P_up = (P_up + P_up.T)/2
        
        return Rot_up, v_up, p_up, b_omega_up, b_acc_up, P_up

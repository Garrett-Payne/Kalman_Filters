import os
import numpy as np
from collections import namedtuple
import datetime
import pickle
import glob
from navpy import lla2ned

# created named tuples to contain specific data fields
OxtsPacket = namedtuple('OxtsPacket',
'lat, lon, alt, ' + 'roll, pitch, yaw, ' + 'vn, ve, vf, vl, vu, '
                                            '' + 'ax, ay, az, af, al, '
                                                'au, ' + 'wx, wy, wz, '
                                                            'wf, wl, wu, '
                                                            '' +
'pos_accuracy, vel_accuracy, ' + 'navstat, numsats, ' + 'posmode, '
                                                        'velmode, '
                                                        'orimode')

OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')


'''def prepare_data(args, dataset, dataset_name, i, idx_start=None, idx_end=None, to_numpy=False):
    # get data
    t, ang_gt, p_gt, v_gt, u = dataset.get_data(dataset_name)

    # get start instant
    if idx_start is None:
        idx_start = 0
    # get end instant
    if idx_end is None:
        idx_end = t.shape[0]

    t = t[idx_start: idx_end]
    u = u[idx_start: idx_end]
    ang_gt = ang_gt[idx_start: idx_end]
    v_gt = v_gt[idx_start: idx_end]
    p_gt = p_gt[idx_start: idx_end] - p_gt[idx_start]

    if to_numpy:
        t = t.cpu().double().numpy()
        u = u.cpu().double().numpy()
        ang_gt = ang_gt.cpu().double().numpy()
        v_gt = v_gt.cpu().double().numpy()
        p_gt = p_gt.cpu().double().numpy()
    return t, ang_gt, p_gt, v_gt, u'''


'''def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)'''



'''def umeyama_alignment(x, y, with_scale=False):
    """
    Computes the least squares solution parameters of an Sim(m) matrix that minimizes the distance between a set of
    registered points.

    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """


    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c'''

def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def normalize_rot(Rot):
    # The SVD is commonly written as a = U S V.H.
    # The v returned by this function is V.H and u = U.
    U, _, V = np.linalg.svd(Rot, full_matrices=False)
    S = np.eye(3)
    S[2, 2] = np.linalg.det(U) * np.linalg.det(V)
    return U.dot(S).dot(V)


def skew(x):
    X = np.array([[0, -x[2], x[1]],
                    [x[2], 0, -x[0]],
                    [-x[1], x[0], 0]])
    return X


def from_rpy(roll, pitch, yaw):
    return rotz(yaw).dot(roty(pitch).dot(rotx(roll)))


def rot_angles_to_matrix_XYZ(x, y, z):
    return rotx(x).dot(roty(y).dot(rotz(z)))


def to_rpy(Rot):
    pitch = np.arctan2(-Rot[2, 0], np.sqrt(Rot[0, 0]**2 + Rot[1, 0]**2))
    if np.isclose(pitch, np.pi / 2.):
        yaw = 0.
        roll = np.arctan2(Rot[0, 1], Rot[1, 1])
    elif np.isclose(pitch, -np.pi / 2.):
        yaw = 0.
        roll = -np.arctan2(Rot[0, 1], Rot[1, 1])
    else:
        sec_pitch = 1. / np.cos(pitch)
        yaw = np.arctan2(Rot[1, 0] * sec_pitch,
                            Rot[0, 0] * sec_pitch)
        roll = np.arctan2(Rot[2, 1] * sec_pitch,
                            Rot[2, 2] * sec_pitch)
    return roll, pitch, yaw


def pose_from_oxts_packet(packet, scale):
    """Helper method to compute a SE(3) pose matrix from an OXTS packet.
    """
    er = 6378137.  # earth radius (approx.) in meters

    # Use a Mercator projection to get the translation vector
    tx = scale * packet.lon * np.pi * er / 180.
    ty = scale * er * np.log(np.tan((90. + packet.lat) * np.pi / 360.))
    tz = packet.alt
    t = np.array([tx, ty, tz])

    # Use the Euler angles to get the rotation matrix
    Rx = rotx(packet.roll)
    Ry = roty(packet.pitch)
    Rz = rotz(packet.yaw)
    R = Rz.dot(Ry.dot(Rx))
    # Combine the translation and rotation into a homogeneous transform
    return R, t


def transform_from_rot_trans(R, t):
    """Transformation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def load_oxts_packets_and_poses(oxts_files):
    """Generator to read OXTS ground truth data.
        Poses are given in an East-North-Up coordinate system
        whose origin is the first GPS position.
    """
    # Scale for Mercator projection (from first lat value)
    scale = None
    # Origin of the global coordinate system (first GPS position)
    origin = None

    oxts = []

    for filename in oxts_files:
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.split()
                # Last five entries are flags and counts
                line[:-5] = [float(x) for x in line[:-5]]
                line[-5:] = [int(float(x)) for x in line[-5:]]

                packet = OxtsPacket(*line)

                if scale is None:
                    scale = np.cos(packet.lat * np.pi / 180.)

                R, t = pose_from_oxts_packet(packet, scale)

                if origin is None:
                    origin = t

                T_w_imu = transform_from_rot_trans(R, t - origin)

                oxts.append(OxtsData(packet, T_w_imu))
    return oxts


def load_timestamps(data_path):
    """Load timestamps from file."""
    timestamp_file = os.path.join(data_path, 'oxts', 'timestamps.txt')

    # Read and parse the timestamps
    timestamps = []
    with open(timestamp_file, 'r') as f:
        for line in f.readlines():
            # NB: datetime only supports microseconds, but KITTI timestamps
            # give nanoseconds, so need to truncate last 4 characters to
            # get rid of \n (counts as 1) and extra 3 digits
            t = datetime.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
            timestamps.append(t)
    return timestamps


def read_raw_oxts_data(oxts_file_loc):
    oxts_files = sorted(glob.glob(os.path.join(oxts_file_loc, 'oxts', 'data', '*.txt')))
    oxts = load_oxts_packets_and_poses(oxts_files)
    lat_oxts = np.zeros(len(oxts))
    lon_oxts = np.zeros(len(oxts))
    alt_oxts = np.zeros(len(oxts))
    roll_oxts = np.zeros(len(oxts))
    pitch_oxts = np.zeros(len(oxts))
    yaw_oxts = np.zeros(len(oxts))
    roll_gt = np.zeros(len(oxts))
    pitch_gt = np.zeros(len(oxts))
    yaw_gt = np.zeros(len(oxts))
    t = load_timestamps(oxts_file_loc)
    acc = np.zeros((len(oxts), 3))
    acc_bis = np.zeros((len(oxts), 3))
    gyro = np.zeros((len(oxts), 3))
    gyro_bis = np.zeros((len(oxts), 3))
    p_gt = np.zeros((len(oxts), 3))
    v_gt = np.zeros((len(oxts), 3))
    v_rob_gt = np.zeros((len(oxts), 3))

    k_max = len(oxts)
    for k in range(k_max):
        oxts_k = oxts[k]
        t[k] = 3600 * t[k].hour + 60 * t[k].minute + t[k].second + t[
            k].microsecond / 1e6
        lat_oxts[k] = oxts_k[0].lat
        lon_oxts[k] = oxts_k[0].lon
        alt_oxts[k] = oxts_k[0].alt
        acc[k, 0] = oxts_k[0].af
        acc[k, 1] = oxts_k[0].al
        acc[k, 2] = oxts_k[0].au
        acc_bis[k, 0] = oxts_k[0].ax
        acc_bis[k, 1] = oxts_k[0].ay
        acc_bis[k, 2] = oxts_k[0].az
        gyro[k, 0] = oxts_k[0].wf
        gyro[k, 1] = oxts_k[0].wl
        gyro[k, 2] = oxts_k[0].wu
        gyro_bis[k, 0] = oxts_k[0].wx
        gyro_bis[k, 1] = oxts_k[0].wy
        gyro_bis[k, 2] = oxts_k[0].wz
        roll_oxts[k] = oxts_k[0].roll
        pitch_oxts[k] = oxts_k[0].pitch
        yaw_oxts[k] = oxts_k[0].yaw
        v_gt[k, 0] = oxts_k[0].ve
        v_gt[k, 1] = oxts_k[0].vn
        v_gt[k, 2] = oxts_k[0].vu
        v_rob_gt[k, 0] = oxts_k[0].vf
        v_rob_gt[k, 1] = oxts_k[0].vl
        v_rob_gt[k, 2] = oxts_k[0].vu
        p_gt[k] = oxts_k[1][:3, 3]
        Rot_gt_k = oxts_k[1][:3, :3]
        roll_gt[k], pitch_gt[k], yaw_gt[k] = to_rpy(Rot_gt_k)


    t = np.array(t) - t[0]
    # some data can have gps out
    if np.max(t[:-1] - t[1:]) > 0.1:
        print(oxts_file_loc + " has time problem", 'yellow')
    ang_gt = np.zeros((roll_gt.shape[0], 3))
    ang_gt[:, 0] = roll_gt
    ang_gt[:, 1] = pitch_gt
    ang_gt[:, 2] = yaw_gt

    p_oxts = lla2ned(lat_oxts, lon_oxts, alt_oxts, lat_oxts[0], lon_oxts[0],
                        alt_oxts[0], latlon_unit='deg', alt_unit='m', model='wgs84')
    p_oxts[:, [0, 1]] = p_oxts[:, [1, 0]]  # see note

    # take correct imu measurements
    u = np.concatenate((gyro_bis, acc_bis), -1)
    return t, ang_gt, p_gt, v_gt, u



def read_parsed_oxts_data(file_loc):
    file_names = os.listdir(file_loc)
    with open(file_loc + file_names[0], "rb") as file_pi:
        pickle_dict = pickle.load(file_pi)
    t = pickle_dict['t'].numpy()
    ang_gt = pickle_dict['ang_gt'].numpy()
    p_gt = pickle_dict['p_gt'].numpy()
    v_gt = pickle_dict['v_gt'].numpy()
    u = pickle_dict['u'].numpy()
    return t, ang_gt, p_gt, v_gt, u



def rotation_integral(phi):
    """Rotation of an angle based upon Rodrigues' rotation formula."""
    return (np.eye(3) + skew(phi))
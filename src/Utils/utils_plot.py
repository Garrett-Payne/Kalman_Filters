
import matplotlib.pyplot as plt
import numpy as np


def plot_route(p_gt, p):
    """Create a 2D map plot that compares filter performance to Ground Truth"""
    plt.figure()
    plt.plot(p_gt[:,0],p_gt[:,1],label='Ground Truth')
    plt.plot(p[:,0],p[:,1],label='EKF')
    plt.xlabel("E/W Direction (m)")
    plt.ylabel("N/S Direction (m)")
    plt.title("Comparison of Filter Results With Truth")
    plt.legend()


def plot_2D_error(t, p_gt, p):
    """Plot calculated 2D (Radial) error over time"""
    radial_error = np.sqrt((p_gt[:,0]- p[:,0])**2 + (p_gt[:,1]-p[:,1])**2)
    plt.figure()
    plt.plot(t,radial_error)
    plt.xlabel("Time (s)")
    plt.ylabel("Radial Error (m)")
    plt.title("Radial (2D) Error Over Time")

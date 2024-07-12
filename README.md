### Introduction

This Repo is intended to show off examples of different versions of Kalman Filters, specifically applied to navigation. The different filters are implemented in python with unique classes that share the same framework for ease of testing and comparison.

First and foremost, this work was built around the great work that Martin Brossard and Liulong Ma did with implementing an IEKF (Invariant Extended Kalman Filter) - refer to their github repo:(https://github.com/mbrossar/ai-imu-dr?tab=MIT-1-ov-file) 
and their IEEE paper:
(https://ieeexplore.ieee.org/document/9035481), [ArXiv paper](https://arxiv.org/pdf/1904.06064.pdf)

While their paper focuses on applying a Neural Network for creating an adaptive noise matrix within the filter, their utility for me was in how the implemented the IEKF in Python. They created a great framework for implementing and testing filters, and I based the different filter classes on their framework for their IEKF class - and within this repo, you can find their implementation of the IEKF. I also borrowed a lot of their code for parsing data since it worked very well.


### Overview

Kalman Filters are a powerful group of algorithms that are used to take in raw, noisy measurements and calculate a cleaner state estimate. Kalman Filters are powerful in that they can be used to calculate states that aren't directly measurable, using data points that are measurable. 

While powerful on their own, basic Kalman filters are only applicable to linear systems. Extended Kalman Filters (EKFs) are used in cases where systems and states are non-linear, which is much more applicable to real-world systems. The Extended Kalman Filter attempts to linearize the system modeled to estimate the states over time.

Kalman Filtering is very important in navigation. Kalman Filters are used for navigation systems to combine measurements from navigation-based sensors and systems to get an estimate of a system's position, velocity, altitude, acceleration, and/or attitude. Kalman Filtering is used inside GPS receivers and Inertial Navigation Systems (INS's), which combine an inertial-based sensor, such as an Inertial Navigation Unit (IMU), with a GPS receiver.

In the implementation of this repo, we're going to test out different versions/applications of Kalman Filters as part of a simplified INS (Inertial Navigation System). Usually INS's integrate GPS/GNSS receivers, but for showing off the filters, we will use data from an IMU only.

The data to use to show off these filters are from the KITTI Vision Benchmark Suite - https://www.cvlibs.net/datasets/kitti/ - which provides IMU data along with other navigation/sensing based data. Brossard, et al used this dataset for their testing, and since this is an open-source database used for testing autonomous vehicle sensing benchmarking, it will be a good source to use for basic testing.

### Concepts

## IMU Data & Relative/Absolute Positioning
The Kalman Filters implemented here are based upon using IMU data - accelerometer and gyroscopic measurements - to update position, velocity, and attitude. IMUs are good for measuring relative changes - acceleration measurements can help you measure changes in position and velocity, and gyroscopic measurements can help you measure your change in attitude. But these are only relative changes, and to be able to get your position on a map, you need to know your starting point. For this implementation, a starting point is assumed to be known - the starting position, velocity, and attitude - so that the absolute states can be updated with the IMU measurements.

## Kalman Filtering Basics
Kalman Filters are complex, and I'd refer readers to a few different resources to learn the technical details about them:
* https://en.wikipedia.org/wiki/Kalman_filter
* https://www.kalmanfilter.net/default.aspx
* https://en.wikipedia.org/wiki/Extended_Kalman_filter

In simple terms, Kalman Filters can estimate _states_, which are variables to describe some physical phenomenon that can be modeled by sets of equations. The Kalman Filters can take in _measurements_, which are data points that can be directly measured or observed by some source - such as an IMU! Kalman Filters also take as inputs estimates of the noise of measurements, and will dynamically weigh the measurements and how they affect the system states over time. Kalman Filters keep track of the system noise in the form of a covariance matrix that gets updated over time along with the states.

Kalman Filters, while complex, can be boiled down to 2 main steps: Predict and Update. These are highlighted in the different code classes, and will be expanded on here:

# #1 Predict
The first part of a Kalman Filter is typically called the Predict Step (sometimes also referred to as propagation, or mechanization, based upon application). This step uses knowledge of how the states update over time to predict what the states will be at the next timestamp. The states are predicted using these equations, and the system noise covariance matrix is propagated as well.

# #2 Update
The second part of a Kalman Filter is called the update step. This step compares the input measurement and predicted state to get a better estimate of the actual state. This step calculates the Kalman Gain - a matrix that holds weights for the measurements and estimated states - and applies it to dynamically update the states.
Updates are based upon innovation and/or residual terms. This compares the input measurement values with the estimated measurement values based upon the predicted state estimates. Innovation values are this difference when the estimated measurement is calculated before the state is updated at the iteration, while residual values are the difference when the estimated measurements are updated at that iteration based upon current information.
The update step typically leverages linear algebra to more efficiently hold important calculation information and values. Vectors and matrices are used and are updated using well-known linear algebra equations. I would recommend some of the resources I've linked for learning more about what's included in these vectors and matrices and how they are updated, since the variables also include various sub/superscripts to detail information about their timestamp and/or what kind of information they hold.



## Code
This implementation is done in Python. The original IEKF repo was developed using Python 3.5, while I have been using Python 3.12. Miles may vary if using earlier versions of Python.
 
### Installation & Prerequisites
    
1.  Install the following required packages, `matplotlib`, `numpy`, `navpy`, e.g. with the pip3 command
```
pip3 install matplotlib numpy navpy
```
    
2.  Clone this repo
```
git clone <insert_library_https_link_here>
```

### Testing
These filters were tested within this framework using data collected from the KITTI Vision Benchmark Test Suite. The process of downloading data, setting up a folder, then running the data is detailed below.


## Data

# Gathering data
On the KITTI Vision Benchmark Suite website, go to the 'Raw Data' tab (https://www.cvlibs.net/datasets/kitti/raw_data.php). NOTE: you will have to create an account with this website to be able to download data. Once you have an account, scroll down on that page to where you see the data categories and data for individual runs. Click on the 'unsynced+unrectified data' link for one of them, and you'll be able to download the data from that specific run.

# Structure data
In this top-level directory, create a new folder called 'data'. Within this folder, copy the downloaded dataset from the KITTI website.
For example, you should within this directory, you should have the structure like:
data/2011_09_30_drive_0028_extract

and within the '2011_09_30_drive_0028_extract' folder, there should be a subfolder named 'oxts' - this is the IMU data that is needed to run the filter.

# Run the filter(s)

Select the 'main_<FILTER>.py' script to run the intended filter. Make sure to update the top variable `oxts_file_loc` to where your data is located.

You should then be able to run the script by running:
`python main_<FILTER>.py`. 



### Citations

Putting another citation here for the work that Brossard et al did on the code and AI-IMU paper.

```
@article{brossard2019aiimu,
  author = {Martin Brossard and Axel Barrau and Silv\`ere Bonnabel},
  journal={IEEE Transactions on Intelligent Vehicles}, 
  title = {{AI-IMU Dead-Reckoning}},
  year = {2020}
}
```

### Authors
Garrett Payne, Senior Navigation Engineer
Bachelor of Aerospace Engineering, Auburn University
Master of Science in Aerospace Systems Engineering, University of Alabama in Huntsville

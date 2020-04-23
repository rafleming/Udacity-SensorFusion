# Udacity Sensor Fusion Nanodegree
Coursework repository for the Udacity SFND.


#
## Overview 

From Udacity: 

*Learn to detect obstacles in lidar point clouds through clustering and segmentation, apply thresholds and filters to radar data in order to accurately track objects, and augment your perception by projecting camera images into three dimensions and fusing these projections with other sensor data. Combine this sensor data with Kalman filters to perceive the world around a vehicle and track objects over time.*


#
## Project 1. Lidar

Process raw lidar data with filtering, segmentation, and clustering to detect other vehicles on the road.

<img src="Lidar/media/ObstacleDetectionFPS.gif" />

#
## Project 2. Camera - 2D Feature Tracking

As part of a collision detection system a stream of camera images are used to develop a feature tracking system and then to assess the performance of various keypoint detector and descriptor combinations.

<img src="Camera/2D Feature Tracking/images/keypoints.png"/>


#
## Project 3. Camera - 3D Object Tracking

Keypoint correspondences are used to match 3D objects over time and then to compute the time-to-collision to the preceding vehicle for both Lidar and Camera.

<img src="Camera/3D Object Tracking/images/sample.gif"/>

#
## Project 4. Radar

Analyze radar signatures to detect and track objects. Calculate velocity and orientation by correcting for radial velocity distortions, noise, and occlusions.

<img src="Radar/images/project-layout.png"/>


#
## Project 5. Kalman Filters

Data from multiple sources used to build Kalman filters, both extended and unscented to track nonlinear movement.

<img src="Unscented Kalman Filters/media/ukf_highway_tracked.gif"/>

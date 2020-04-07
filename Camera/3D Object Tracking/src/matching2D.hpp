
#ifndef matching2D_hpp
#define matching2D_hpp

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"

// Detect keypoints using the given detector type. Supports SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE and SIFT.
double detectKeypoints(const std::string& detectorType, std::vector<cv::KeyPoint>& keypoints, cv::Mat& imgGray, bool bVis = false);

double descKeypoints(const std::string& descriptorType, std::vector<cv::KeyPoint>& keypoints, cv::Mat& img, cv::Mat& descriptors);

void matchDescriptors(std::vector<cv::KeyPoint>& kPtsSource, std::vector<cv::KeyPoint>& kPtsRef, cv::Mat& descSource, cv::Mat& descRef,
                      std::vector<cv::DMatch>& matches, std::string descriptorType, std::string matcherType, std::string selectorType);

#endif /* matching2D_hpp */

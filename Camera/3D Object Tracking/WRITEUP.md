# SFND 3D Object Tracking

<img src="images/sample.gif" />


## Final Project Write-Up
#

### FP.0 Final Report
*Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf.*

Here.

#
### FP.1 Match 3D Objects

*Implement the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property). Matches must be the ones with the highest number of keypoint correspondences.*

camFusion_Student.cpp
```cpp
void matchBoundingBoxes(const DataFrame& prevFrame, const DataFrame& currFrame, const std::vector<cv::DMatch>& matches, std::map<int, int>& bbBestMatches)
{
    std::map<std::pair<int, int>, int> pointCounts;

    for (const auto& match : matches)
    {     
        // find bb ids for the previous frame keypoint
        const cv::KeyPoint& prevKeypoint = prevFrame.keypoints[match.queryIdx];
       
        std::vector<int> prevFrameBBIds;
        for (int i = 0; i < prevFrame.boundingBoxes.size(); i++)
        {
            if (prevFrame.boundingBoxes[i].roi.contains(prevKeypoint.pt))
            {
                prevFrameBBIds.push_back(i);
            }
        }

        // find bb ids for the current frame keypoint
        const cv::KeyPoint& currKeypoint = currFrame.keypoints[match.trainIdx];
     
        std::vector<int> currFrameBBIds;
        for (int i = 0; i < currFrame.boundingBoxes.size(); i++)
        {
            if (currFrame.boundingBoxes[i].roi.contains(currKeypoint.pt))
            {
                currFrameBBIds.push_back(i);
            }
        }

        // keep track of the point counts for BB matches
        for (auto prevBBId : prevFrameBBIds)
        {
            for (auto currBBId : currFrameBBIds)
            {
                auto key = std::make_pair(prevBBId, currBBId);

                if (pointCounts.count(key))
                {
                    pointCounts[key] = pointCounts[key] + 1;
                }
                else
                    pointCounts[key] = 1;
            }
        }
    }

    // finally find the best matches based on the point counts
    for (int i = 0; i < prevFrame.boundingBoxes.size(); i++)
    {
        int maxPoints       = 0;
        int currFrameBBIdx  = 0;

        for (int j = 0; j < currFrame.boundingBoxes.size(); j++)
        {
            auto key = std::make_pair(i, j);

            if (pointCounts.count(key) && 
                pointCounts[key] > maxPoints)
            {
                maxPoints = pointCounts[key];
                currFrameBBIdx = j;
            }
        }

        // store the best match
        bbBestMatches[i] = currFrameBBIdx;
    }
}
```

#
### FP.2 Compute Lidar-based TTC
*Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame.*

camFusion_Student.cpp
```cpp
void computeTTCLidar(std::vector<LidarPoint>& lidarPointsPrev, 
                     std::vector<LidarPoint>& lidarPointsCurr, double frameRate, double& TTC)
{
    // start by getting the x values of the points in both the previous and current frame
    std::vector<float> prevPointX;
    std::vector<float> currPointX;

    for (const auto& p : lidarPointsPrev)
    {
        prevPointX.push_back(p.x);
    }

    for (const auto& p : lidarPointsCurr)
    {
        currPointX.push_back(p.x);
    }

    // avoid outliers by finding the mean of the lowest 10 x values (using std::nth_element avoids full sort)
    const int samplePoints = 10;

    float prevSampleMeanX = 0.0;
    float currSampleMeanX = 0.0;

    // prev points
    if (prevPointX.size() > samplePoints)
    {
        std::vector<float>::iterator b = prevPointX.begin();
        std::vector<float>::iterator e = prevPointX.end();
        std::vector<float>::iterator s = b;

        std::advance(s, samplePoints);
        std::nth_element(b, s, e);

        prevSampleMeanX = std::accumulate(prevPointX.begin(), s, 0.0) / samplePoints;
    }
    else
    {
        prevSampleMeanX = std::accumulate(prevPointX.begin(), prevPointX.end(), 0.0) / prevPointX.size();
    }

    // curr points
    if (currPointX.size() > samplePoints)
    {
        std::vector<float>::iterator b = currPointX.begin();
        std::vector<float>::iterator e = currPointX.end();
        std::vector<float>::iterator s = b;

        std::advance(s, samplePoints);
        std::nth_element(b, s, e);

        currSampleMeanX = std::accumulate(currPointX.begin(), s, 0.0) / samplePoints;
    }
    else
    {
        currSampleMeanX = std::accumulate(currPointX.begin(), currPointX.end(), 0.0) / currPointX.size();
    }
    
  
    // finally set the TTC
    float dt = 1 / frameRate;
    TTC = currSampleMeanX * dt / (prevSampleMeanX - currSampleMeanX);
}
```


#
### FP.3 Associate Keypoint Correspondences with Bounding Boxes
*Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box.*

camFusion_Student.cpp
```cpp
// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox& boundingBox, std::vector<cv::KeyPoint>& kptsPrev, std::vector<cv::KeyPoint>& kptsCurr, std::vector<cv::DMatch>& kptMatches)
{
    // First get the Euclidean distances
    std::multiset<double> distances;
    for (const cv::DMatch& match : kptMatches)
    {
        const cv::KeyPoint& keypointCurr = kptsCurr[match.trainIdx];

        if (boundingBox.roi.contains(keypointCurr.pt))
        {
            const cv::KeyPoint& keypointPrev = kptsPrev[match.queryIdx];
            distances.emplace(cv::norm(keypointCurr.pt - keypointPrev.pt));
        }
    }

    // Find the mean
    const double meanEuclideanDist =
        std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size();

    // Handle outliers by filtering the 10% of points that deviate furthest from the mean     
    const double filterOutliersFactor = 0.1;

    auto distancesIt = distances.crend();
    std::advance(distancesIt, round(filterOutliersFactor * distances.size()));

    double distanceFilterThreshold = *distancesIt;

    for (const cv::DMatch& match : kptMatches)
    {
        const cv::KeyPoint& keypointCurr = kptsCurr[match.trainIdx];

        if (boundingBox.roi.contains(keypointCurr.pt))
        {
            const auto& prevKpt = kptsPrev[match.queryIdx];
            const double euclideanDistance = cv::norm(keypointCurr.pt - prevKpt.pt);

            if (euclideanDistance <= distanceFilterThreshold)
            {
                boundingBox.keypoints.push_back(keypointCurr);
                boundingBox.kptMatches.push_back(match);
            }
        }
    }
}
```

#
### FP.4 Compute Camera-based TTC
*Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame.*


```cpp
// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint>& kptsPrev, 
                      std::vector<cv::KeyPoint>& kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double& TTC, cv::Mat* visImg)
{
    // first calculate the distance ratios between all matched keypoints
    std::vector<double> distRatios; 
    double minDist = 100.0; // min. required distance

    for (const cv::DMatch& match1 : kptMatches)
    { 
        const cv::KeyPoint& currKeypoint1 = kptsCurr.at(match1.trainIdx);
        const cv::KeyPoint& prevKeypoint1 = kptsPrev.at(match1.queryIdx);

        for (const cv::DMatch& match2 : kptMatches)
        { 
            const cv::KeyPoint& currKeypoint2 = kptsCurr.at(match2.trainIdx);
            const cv::KeyPoint& prevKeypoint2 = kptsPrev.at(match2.queryIdx);

            // compute distances and distance ratios
            double currDistance = cv::norm(currKeypoint1.pt - currKeypoint2.pt);
            double prevDistance = cv::norm(prevKeypoint1.pt - prevKeypoint2.pt);

            if (currDistance >= minDist && prevDistance > 0)
            { 
                double ratio = currDistance / prevDistance;
                distRatios.push_back(ratio);
            }
        } 
    }     

    if (!distRatios.empty())
    {
        std::sort(distRatios.begin(), distRatios.end());
        long medIndex = floor(distRatios.size() / 2.0);
        // compute median dist. ratio to remove outlier influence
        double medDistRatio = distRatios.size() % 2 == 0 ?
            (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex];

        // finally set the TTC
        double dt = 1 / frameRate;
        TTC = -dt / (1 - medDistRatio);
    }
    else
    {
        // we're unable to calculate TTC if the distance ratios are empty
        TTC = NAN;
    }
}

```



#
### FP.5 Performance Evaluation 1
*Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened.*


#### Example 1.  
One example of why the lidar only TTC calculation appears to be incorrect can be found by examining the top-view lidar images between frames 5 and 6 as shown below.

<img src="images/Lidar/lidar-top-view-5.png" />
frame 5.

<img src="images/Lidar/lidar-top-view-6.png" />
frame 6.

The lidar TTC at frame 6 I believe is lower than expected due to the outliers on the preceding frame 5 (these can be observed above the 2m line distance marker), resulting in a higher mean X value for frame 5. 

As these points are not observed in frame 6, the resulting TTC is lower than expected at 7.49s.   

<img src="images/Camera/camera-6.png" />

#### Example 2.  

A further example of a lidar TTC calculation that does not seem plausible can be found between frames 6 and 7.

<img src="images/Lidar/lidar-top-view-6.png" />
frame 6.

<img src="images/Lidar/lidar-top-view-7.png" />
frame 7.

In frame 7 the lidar appears to have captured points on the left-rear wheel arch of the preceding vehicle that were not captured in frame 6. This results in a larger mean x distance in frame 7 than would have been calculated otherwise which in turn results in a much higher TTC of 42.97s.

<img src="images/Camera/camera-7.png" />


#
### FP.6 Performance Evaluation 2
*Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.*

From the mid-term project the top 3 detector/descriptor combinations were identfied. These have been used here in order to compare the differences in TTC for each for both lidar and camera calculations and in doing so it was possible to identify cases where the camera-based TTC estimation was inconsistent.

|     Place      |            Combination                     | 
|  ------------  |           -------------                    |
|   1st    | FAST + BRIEF | 
|   2nd    | FAST + BRISK |    
|   3nd    | FAST + ORB | 

#### 1. FAST + BRIEF

|     Frame      |            LiDAR TTC (s)    | Camera TTC (s) |
|  ------------  |           -------------   |   -------------   |
|1| 13.60 | 11.78
|2| 11.66 | 12.19
|3| 17.46 | 13.77
|4| 11.62 | 13.20
|5| 13.31 | **22.29**
|6| 7.49 | 13.25
|7| 42.98 | 12.09
|8| 17.87 | 12.31
|9| 13.91 | 13.39
|10| 14.92 | 13.67
|11| 8.85 | 14.11
|12| 12.43 | 11.70
|13| 9.16 | 12.09
|14| 11.36 | 11.68
|15| 7.99 | 11.91
|16| 7.63 | 12.68
|17| 11.98 | 8.62
|18| 8.86 | 11.96


#### 2. FAST + BRISK

|     Frame      |            LiDAR TTC (s)    | Camera TTC (s) |
|  ------------  |           -------------   |   -------------   |
|1| 13.60 | 12.53
|2| 11.66 | 12.55
|3| 17.46 | 14.18
|4| 11.62 | 13.28
|5| 13.31 | **inf**
|6| 7.49 | 13.34
|7| 42.98 | 11.96
|8| 17.87 | 11.89
|9| 13.91 | 12.11
|10| 14.92 | 13.74
|11| 8.85 | 13.60
|12| 12.43 | 12.71
|13| 9.16 | 12.48
|14| 11.36 | 11.60
|15| 7.99 | 12.26
|16| 7.63 | 12.79
|17| 11.98 | 9.93
|18| 8.86 | 11.96


#### 3. FAST + ORB

|     Frame      |            LiDAR TTC (s)    | Camera TTC (s) |
|  ------------  |           -------------   |   -------------   |
|1| 13.60 | 12.01
|2| 11.66 | 11.52
|3| 17.46 | 14.41
|4| 11.62 | 13.96
|5| 13.31 | **159.61**
|6| 7.49 | 13.40
|7| 42.98 | 12.84
|8| 17.87 | 12.60
|9| 13.91 | 13.35
|10| 14.92 | 14.42
|11| 8.85 | 17.82
|12| 12.43 | 11.70
|13| 9.16 | 12.64
|14| 11.36 | 11.60
|15| 7.99 | 11.49
|16| 7.63 | 11.89
|17| 11.98 | 10.66
|18| 8.86 | 12.60


From the results it can be observed than in frame 5 the camera-based TTC estimate is inconsistent across each of the detector/descriptor combinations. In particular the INF value when using the FAST + BRISK combinations is a result of the condition:

```cpp
if (!distRatios.empty())
{
    ...
}
else
{
    // we're unable to calculate TTC if the distance ratios are empty
    TTC = NAN;
}
```

It was noted in this case that this was due to the computed distance ratios not meeting the minumum required distance.
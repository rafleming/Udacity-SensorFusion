#include <iostream>
#include <algorithm>
#include <numeric>
#include <list>
#include <set>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox>& boundingBoxes, std::vector<LidarPoint>& lidarPoints, float shrinkFactor, cv::Mat& P_rect_xx, cv::Mat& R_rect_xx, cv::Mat& RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;

        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox>& boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
        float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, cv::WINDOW_KEEPRATIO);
    cv::imshow(windowName, topviewImg);

    if (bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


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
# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />


## Mid-Term Project Write-Up
#

### MP.0 Mid-Term Report
*Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf.*

Here.

#
### MP.1 Data Buffer Optimization

*Implement a vector for dataBuffer objects whose size does not exceed a limit (e.g. 2 elements). This can be achieved by pushing in new elements on one end and removing elements on the other end.*

MidTermProject_Camera_Student.cpp
```cpp
int dataBufferSize = 2;      
    
// list of data frames which are held in memory at the same time
boost::circular_buffer<DataFrame> circularDataBuffer;
circularDataBuffer.set_capacity(dataBufferSize);
```

#
### MP.2 Keypoint Detection
*Implement detectors HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT and make them selectable by setting a string accordingly.*

MidTermProject_Camera_Student.cpp
```cpp
double t = detectKeypoints(detectorType, keypoints, imgGray, false);
```

matching2D_Student.cpp
```cpp
double detectKeypoints(const std::string& detectorType, vector<cv::KeyPoint>& keypoints, cv::Mat& imgGray, bool bVis)
{
    double t = 0;

    if (detectorType.compare("SHITOMASI") == 0)
    {
        t = detectKeypointsShiTomasi(keypoints, imgGray);
    }
    else if (detectorType.compare("HARRIS") == 0)
    {
        t = detectKeypointsHarris(keypoints, imgGray);
    }
    else 
    {
        t = detectKeypointsModern(keypoints, imgGray, detectorType, bVis);
    }

    ...

    return t;
}
```

```cpp
// Detect keypoints in image using the traditional Harris detector
double detectKeypointsHarris(std::vector<cv::KeyPoint>& keypoints, cv::Mat& img)
{
    double t = (double)cv::getTickCount();

    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse)
            { // only store points above a threshold

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {                      // if overlap is >t AND response is higher for new kpt
                            *it = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap)
                {                                     // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        } // eof loop over cols
    }     // eof loop over rows

    return ((double)cv::getTickCount() - t) / cv::getTickFrequency();
}

double detectKeypointsModern(std::vector<cv::KeyPoint>& keypoints, cv::Mat& img, std::string detectorType, bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector;

    if (detectorType.compare("FAST") == 0)
    {
        // difference between intensity of the central pixel and pixels of a circle around this pixel
        int threshold = 30;
        detector = cv::FastFeatureDetector::create(threshold);
    }
    else if (detectorType.compare("BRISK") == 0)
    {
        detector = cv::BRISK::create();
    }
    else if (detectorType.compare("ORB") == 0)
    {
        detector = cv::ORB::create();
    }
    else if (detectorType.compare("AKAZE") == 0)
    {
        detector = cv::AKAZE::create();
    }
    else if (detectorType.compare("SIFT") == 0)
    {
        detector = cv::xfeatures2d::SIFT::create();
    }
    else
    {
        std::cout << "Unsupported detector type " + detectorType;
        throw "Unsupported detector type " + detectorType;
    }

    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    return ((double)cv::getTickCount() - t) / cv::getTickFrequency();
}
```
#
### MP.3 Keypoint Removal
*Remove all keypoints outside of a pre-defined rectangle and only use the keypoints within the rectangle for further processing.*

MidTermProject_Camera_Student.cpp

```cpp
// only keep keypoints on the preceding vehicle
bool bFocusOnVehicle = true;

if (bFocusOnVehicle)
{
    cv::Rect vehicleRect(535, 180, 180, 150);
    std::vector<cv::KeyPoint> vehicleKeypoints;

    for (cv::KeyPoint keypoint : keypoints)
    {
        if (vehicleRect.contains(keypoint.pt))
        {
            vehicleKeypoints.push_back(keypoint);
        }
    }
    keypoints = vehicleKeypoints;
}
```

#
### MP.4 Keypoint Descriptors
*Implement descriptors BRIEF, ORB, FREAK, AKAZE and SIFT and make them selectable by setting a string accordingly.*

MidTermProject_Camera_Student.cpp
```cpp
string descriptorType = "BRIEF"; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
t = descKeypoints(descriptorType, (circularDataBuffer.end() - 1)->keypoints, (circularDataBuffer.end() - 1)->cameraImg, descriptors);
cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
```

matching2D_Student.cpp
```cpp
// Use one of several types of state-of-art descriptors to uniquely identify keypoints
double descKeypoints(const string& descriptorType, vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;

    if (descriptorType.compare("BRISK") == 0)
    {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if(descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }
    else
    {
        std::cout << "Unsupported descriptor type " + descriptorType;
        throw "Unsupported descriptor type " + descriptorType;
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    return ((double)cv::getTickCount() - t) / cv::getTickFrequency();
}
```

#
### MP.5 Descriptor Matching
*Implement FLANN matching as well as k-nearest neighbor selection. Both methods must be selectable using the respective strings in the main function.*

matching2D_Student.cpp
```cpp
if (matcherType.compare("MAT_BF") == 0)
{
    int normType = cv::NORM_HAMMING;
    matcher = cv::BFMatcher::create(normType, crossCheck);
}
else if (matcherType.compare("MAT_FLANN") == 0)
{
    if (descSource.type() != CV_32F)
    { 
        // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
        descSource.convertTo(descSource, CV_32F);
        descRef.convertTo(descRef, CV_32F);
    }

    matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
}
```

#
### MMP.6 Descriptor Distance Ratio
*Use the K-Nearest-Neighbor matching to implement the descriptor distance ratio test, which looks at the ratio of best vs. second-best match to decide whether to keep an associated pair of keypoints.*

matching2D_Student.cpp
```cpp
// k nearest neighbors (k=2)
vector<vector<cv::DMatch>> knn_matches;
matcher->knnMatch(descSource, descRef, knn_matches, 2);

// filter matches using descriptor distance ratio test
double minDescDistRatio = 0.8;

for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
{
    if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
    {
        matches.push_back((*it)[0]);
    }
}
```

#
### MP.7 Performance Evaluation 1
*Count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size. Do this for all the detectors you have implemented.*

See [results spreadsheet](Results.xlsx).

#
### MP.8 Performance Evaluation 2
*Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.*

See [results spreadsheet](Results.xlsx).

#
### MP.9 Performance Evaluation 3
*Log the time it takes for keypoint detection and descriptor extraction. The results must be entered into a spreadsheet and based on this data, the TOP3 detector / descriptor combinations must be recommended as the best choice for our purpose of detecting keypoints on vehicles.*

See [results spreadsheet](Results.xlsx).
/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>

#include <boost/circular_buffer.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../../../";
 
    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // no. of images which are held in memory (ring buffer) at the same time
    int dataBufferSize = 2;      
    
    // list of data frames which are held in memory at the same time
    boost::circular_buffer<DataFrame> circularDataBuffer;
    circularDataBuffer.set_capacity(dataBufferSize);

    // visualize results
    bool bVis = false;            

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;

        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        circularDataBuffer.push_back(frame);

        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        string detectorType = "FAST";   // SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        
        double t = detectKeypoints(detectorType, keypoints, imgGray, false);
        std::cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

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

        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            {
                // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }

            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (circularDataBuffer.end() - 1)->keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType

        cv::Mat descriptors;
        string descriptorType = "BRIEF"; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT

        t = descKeypoints(descriptorType, (circularDataBuffer.end() - 1)->keypoints, (circularDataBuffer.end() - 1)->cameraImg, descriptors);
        cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;

        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (circularDataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        // wait until at least two images have been processed
        if (circularDataBuffer.size() > 1)
        {
            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType      = "MAT_BF";         // MAT_BF, MAT_FLANN
            string descriptorType   = "DES_BINARY";     // DES_BINARY, DES_HOG
            string selectorType     = "SEL_KNN";        // SEL_NN, SEL_KNN

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            matchDescriptors(
                (circularDataBuffer.end() - 2)->keypoints,   (circularDataBuffer.end() - 1)->keypoints,
                (circularDataBuffer.end() - 2)->descriptors, (circularDataBuffer.end() - 1)->descriptors,
                matches, descriptorType, matcherType, selectorType);

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (circularDataBuffer.end() - 1)->kptMatches = matches;

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            if (bVis)
            {
                cv::Mat cameraImg1 = (circularDataBuffer.end() - 1)->cameraImg;
                cv::Mat cameraImg2 = (circularDataBuffer.end() - 2)->cameraImg;
                cv::Mat matchImg = cameraImg1.clone();

                auto keypoints1 = (circularDataBuffer.end() - 1)->keypoints;
                auto keypoints2 = (circularDataBuffer.end() - 2)->keypoints;

                cv::drawMatches(cameraImg2, keypoints2,
                    cameraImg1, keypoints1,
                    matches, matchImg,
                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                    vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
        }

    } // eof loop over all images

    return 0;
}

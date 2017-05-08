#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/nonfree.hpp>
using namespace cv;
int main(int argc, char** argv)
{
    if(argc < 3) return 0;
    Mat img01=imread(argv[1]);
    Mat img02=imread(argv[2]);
    
    SiftFeatureDetector detector;
    vector<KeyPoint> keypoint01,keypoint02;
    detector.detect(img01,keypoint01);
    detector.detect(img02,keypoint02);


    SiftDescriptorExtractor extractor;
    Mat descriptor01,descriptor02;
    extractor.compute(img01,keypoint01,descriptor01);
    extractor.compute(img02,keypoint02,descriptor02);

    BruteForceMatcher<L2<float> > matcher;
    vector<DMatch> matches;
//    Mat img_matches;
    matcher.match(descriptor01,descriptor02,matches);
//    drawMatches(img01,keypoint01,img02,keypoint02,matches,img_matches);
    
    vector<KeyPoint> R_keypoint01,R_keypoint02;
    for (size_t i=0;i<matches.size();i++)   
    {
        R_keypoint01.push_back(keypoint01[matches[i].queryIdx]);
        R_keypoint02.push_back(keypoint02[matches[i].trainIdx]);
    }

    vector<Point2f>p01,p02;
    for (size_t i=0;i<matches.size();i++)
    {
        p01.push_back(R_keypoint01[i].pt);
        p02.push_back(R_keypoint02[i].pt);
    }

    vector<uchar> RansacStatus;
//    Mat Fundamental= findFundamentalMat(p01,p02,RansacStatus,FM_RANSAC);

    Mat m_Homo = findHomography(p01, p02, CV_RANSAC, 3, RansacStatus);

    vector<Point2f> obj_corners(4);  
    obj_corners[0] = cvPoint(0.0, 0.0);
    obj_corners[1] = cvPoint(0.0, img01.rows);
    obj_corners[2] = cvPoint(img01.cols, img01.rows);
    obj_corners[3] = cvPoint(img01.cols, 0.0);
    vector<Point2f> scene_corners(4);  
    perspectiveTransform(obj_corners, scene_corners, m_Homo);  
    line(img02, scene_corners[0], scene_corners[1], Scalar(0, 0, 255), 2);  
    line(img02, scene_corners[1], scene_corners[2], Scalar(0, 0, 255), 2);  
    line(img02, scene_corners[2], scene_corners[3], Scalar(0, 0, 255), 2);  
    line(img02, scene_corners[3], scene_corners[0], Scalar(0, 0, 255), 2);  
    imshow("消除误匹配点后",img02);

//    vector<KeyPoint> RR_keypoint01,RR_keypoint02;
//    vector<DMatch> RR_matches;
//    int index=0;
//    for (size_t i=0;i<matches.size();i++)
//    {
//        if (RansacStatus[i]!=0)
//        {
//            RR_keypoint01.push_back(R_keypoint01[i]);
//            RR_keypoint02.push_back(R_keypoint02[i]);
//            matches[i].queryIdx=index;
//            matches[i].trainIdx=index;
//            RR_matches.push_back(matches[i]);
//            index++;
//        }
//    }
//    Mat img_RR_matches;
//    drawMatches(img01,RR_keypoint01,img02,RR_keypoint02,RR_matches,img_RR_matches);
//    imshow("消除误匹配点后",img_RR_matches);
//    printf("%lu %lu\n", matches.size(), RR_matches.size());
    waitKey();
}

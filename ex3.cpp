#include <iostream>  
#include <cstdio>  
#include <opencv2/opencv.hpp>  
#include <opencv2/objdetect/objdetect.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/nonfree.hpp>
using namespace std;
using namespace cv; 

String cascadeName = "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml";  

IplImage* cutImage(IplImage* src, CvRect rect) {  
    cvSetImageROI(src, rect);  
    IplImage* dst = cvCreateImage(cvSize(rect.width, rect.height),  
            src->depth,  
            src->nChannels);  
  
    cvCopy(src,dst,0);  
    cvResetImageROI(src);  
    return dst;  
}  

IplImage* detect( Mat& img, CascadeClassifier& cascade, double scale)  
{  
    int i = 0;  
    double t = 0;  
    vector<Rect> faces;  
    Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );  
  
    cvtColor( img, gray, CV_BGR2GRAY );  
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );  
    equalizeHist( smallImg, smallImg );  
  
    t = (double)cvGetTickCount();  
    cascade.detectMultiScale( smallImg, faces,  
        1.3, 2, CV_HAAR_SCALE_IMAGE,  
        Size(30, 30) );  
    t = (double)cvGetTickCount() - t;  
    IplImage iplimg = img;
    IplImage *tmp = cvCloneImage(&iplimg); 
    for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )  
    {  
        IplImage* temp = cutImage(tmp, cvRect(r->x, r->y, r->width, r->height));  
        return temp;  
    }  
  
    return tmp;  
}  

int HistogramBins = 256;  
float HistogramRange1[2]={0,255};  
float *HistogramRange[1]={&HistogramRange1[0]};  
double CompareHist(IplImage* image1, IplImage* image2)  
{  
    IplImage* srcImage;  
    IplImage* targetImage;  
    if (image1->nChannels != 1) {  
        srcImage = cvCreateImage(cvSize(image1->width, image1->height), image1->depth, 1);  
        cvCvtColor(image1, srcImage, CV_BGR2GRAY);  
    } else {  
        srcImage = image1;  
    }  
  
    if (image2->nChannels != 1) {  
        targetImage = cvCreateImage(cvSize(image2->width, image2->height), srcImage->depth, 1);  
        cvCvtColor(image2, targetImage, CV_BGR2GRAY);  
    } else {  
        targetImage = image2;  
    }  
  
    CvHistogram *Histogram1 = cvCreateHist(1, &HistogramBins, CV_HIST_ARRAY,HistogramRange);  
    CvHistogram *Histogram2 = cvCreateHist(1, &HistogramBins, CV_HIST_ARRAY,HistogramRange);  
  
    cvCalcHist(&srcImage, Histogram1);  
    cvCalcHist(&targetImage, Histogram2);  
  
    cvNormalizeHist(Histogram1, 1);  
    cvNormalizeHist(Histogram2, 1);  
  
//    fprintf(stderr, "CV_COMP_CORREL : %.4f\n", cvCompareHist(Histogram1, Histogram2, CV_COMP_CORREL));  
//    fprintf(stderr, "CV_COMP_INTERSECT : %.4f\n", cvCompareHist(Histogram1, Histogram2, CV_COMP_INTERSECT)); 

    double t = cvCompareHist(Histogram1, Histogram2, CV_COMP_CORREL) + cvCompareHist(Histogram1, Histogram2, CV_COMP_INTERSECT);// - cvCompareHist(Histogram1, Histogram2, CV_COMP_BHATTACHARYYA);
  
    cvReleaseHist(&Histogram1);  
    cvReleaseHist(&Histogram2);  
    if (image1->nChannels != 1) {  
        cvReleaseImage(&srcImage);  
    }  
    if (image2->nChannels != 1) {  
        cvReleaseImage(&targetImage);  
    }  
    return t;  
}

double CompareLuv(IplImage* image1, IplImage* image2, int type)  
{  
    IplImage* srcImage;  
    IplImage* targetImage;  
    if (image1->nChannels != 1) {  
        srcImage = cvCreateImage(cvSize(image1->width, image1->height), image1->depth, 3);  
        cvCvtColor(image1, srcImage, CV_BGR2Luv); 
        int step = image1->widthStep;  
        int chanel = image1->nChannels;  
        IplImage * b = cvCreateImage(cvGetSize(image1),IPL_DEPTH_8U,1);
        char * bdata = b->imageData, * data = image1->imageData;  
        for(int i=0;i<image1->height;i++)  
            for(int j=0;j<image1->width;j++) 
            {  
                bdata[i*step/3 + j] = data[i*step + j*chanel + type];  
            }
        srcImage = b; 
    } else {  
        srcImage = image1;  
    }  
  
    if (image2->nChannels != 1) {  
        targetImage = cvCreateImage(cvSize(image2->width, image2->height), srcImage->depth, 3);  
        cvCvtColor(image2, targetImage, CV_BGR2Luv);  
        int step = image2->widthStep;  
        int chanel = image2->nChannels;  
        IplImage * b = cvCreateImage(cvGetSize(image2),IPL_DEPTH_8U,1);
        char * bdata = b->imageData, * data = image2->imageData;  
        for(int i=0;i<image2->height;i++)  
            for(int j=0;j<image2->width;j++) 
            {  
                bdata[i*step/3 + j] = data[i*step + j*chanel + type];  
            }
        targetImage = b; 
    } else {  
        targetImage = image2;  
    }  
  
    CvHistogram *Histogram1 = cvCreateHist(1, &HistogramBins, CV_HIST_ARRAY,HistogramRange);  
    CvHistogram *Histogram2 = cvCreateHist(1, &HistogramBins, CV_HIST_ARRAY,HistogramRange);  
  
    cvCalcHist(&srcImage, Histogram1);  
    cvCalcHist(&targetImage, Histogram2);  
  
    cvNormalizeHist(Histogram1, 1);  
    cvNormalizeHist(Histogram2, 1);  
  
//    fprintf(stderr, "CV_COMP_CORREL : %.4f\n", cvCompareHist(Histogram1, Histogram2, CV_COMP_CORREL));  
//    fprintf(stderr, "CV_COMP_INTERSECT : %.4f\n", cvCompareHist(Histogram1, Histogram2, CV_COMP_INTERSECT)); 

    double t = cvCompareHist(Histogram1, Histogram2, CV_COMP_CORREL) + cvCompareHist(Histogram1, Histogram2, CV_COMP_INTERSECT);// - cvCompareHist(Histogram1, Histogram2, CV_COMP_BHATTACHARYYA);
  
    cvReleaseHist(&Histogram1);  
    cvReleaseHist(&Histogram2);  
    if (image1->nChannels != 1) {  
        cvReleaseImage(&srcImage);  
    }  
    if (image2->nChannels != 1) {  
        cvReleaseImage(&targetImage);  
    }  
    return t;  
}  

bool checkface(Mat srcImg, Mat targetImg)
{ 
    CascadeClassifier cascade;  
    if( !cascade.load( cascadeName ) )  
    {  
        return -1;  
    }  
    IplImage* faceImage1;  
    IplImage* faceImage2;  
    faceImage1 = detect(srcImg, cascade, 1);  
    if (faceImage1 == NULL) {  
        return -1;  
    }  
    faceImage2 = detect(targetImg, cascade, 1);  
    if (faceImage2 == NULL) {  
        return -1;  
    }

    bool status = (CompareLuv(faceImage1, faceImage2, 0) < 1.41 || CompareLuv(faceImage1, faceImage2, 2) < 1.43 || CompareHist(faceImage1, faceImage2) < 1.53);
//    if(CompareLuv(faceImage1, faceImage2, 0) < 1.41 || CompareLuv(faceImage1, faceImage2, 2) < 1.43 || CompareHist(faceImage1, faceImage2) < 1.454) return 0; 
//    
//    printf("%s %s %lf %lf %lf\n", argv[1], argv[2], CompareLuv(faceImage1, faceImage2, 0), CompareLuv(faceImage1, faceImage2, 2), CompareHist(faceImage1, faceImage2)); 
//    
//    namedWindow("image1");  
//    namedWindow("image2");  
//    imshow("image1", Mat(faceImage1));  
//    imshow("image2", Mat(faceImage2));  
//    waitKey();

    cvReleaseImage(&faceImage1);  
    cvReleaseImage(&faceImage2);  
    return !status;  
}

void draw(Mat srcImg, Mat targetImg)
{
    SiftFeatureDetector detector;
    vector<KeyPoint> keypoint01,keypoint02;
    detector.detect(srcImg,keypoint01);
    detector.detect(targetImg,keypoint02);

    SiftDescriptorExtractor extractor;
    Mat descriptor01,descriptor02;
    extractor.compute(srcImg,keypoint01,descriptor01);
    extractor.compute(targetImg,keypoint02,descriptor02);

    BruteForceMatcher<L2<float> > matcher;
    vector<DMatch> matches;
//    Mat img_matches;
    matcher.match(descriptor01,descriptor02,matches);
//    drawMatches(srcImg,keypoint01,targetImg,keypoint02,matches,img_matches);
    
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
    obj_corners[1] = cvPoint(0.0, srcImg.rows);
    obj_corners[2] = cvPoint(srcImg.cols, srcImg.rows);
    obj_corners[3] = cvPoint(srcImg.cols, 0.0);
    vector<Point2f> scene_corners(4);  
    perspectiveTransform(obj_corners, scene_corners, m_Homo);  
    line(targetImg, scene_corners[0], scene_corners[1], Scalar(0, 0, 255), 2);  
    line(targetImg, scene_corners[1], scene_corners[2], Scalar(0, 0, 255), 2);  
    line(targetImg, scene_corners[2], scene_corners[3], Scalar(0, 0, 255), 2);  
    line(targetImg, scene_corners[3], scene_corners[0], Scalar(0, 0, 255), 2);  
//    imshow("消除误匹配点后",targetImg);

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
//    drawMatches(srcImg,RR_keypoint01,targetImg,RR_keypoint02,RR_matches,img_RR_matches);
//    imshow("消除误匹配点后",img_RR_matches);
//    printf("%lu %lu\n", matches.size(), RR_matches.size());
//    waitKey();
}

Mat mergeImg(Mat src1, Mat src2)  
{  
    Mat dst;
    int rows = src1.rows;  
    int cols = src1.cols+src2.cols;  
    dst.create (rows,cols,src1.type ());  
    src1.copyTo (dst(Rect(0,0,src1.cols,src1.rows)));  
    src2.copyTo (dst(Rect(src1.cols,0,src2.cols,src2.rows)));  
    return dst;
}  

int main()
{
    const int leftn = 17, rightn = 18;
    const char *leftI = "left.jpeg", *rightI = "right.jpeg";
    const char *leftF = "image/l%d.png", *rightF = "image/r%d.png";
    for(int i=1;i<=rightn;i++)
    {
        namedWindow("result", CV_WINDOW_NORMAL);  
        Mat left = imread(leftI), right = imread(rightI);
        char tmp[1024];
        sprintf(tmp, rightF, i);
        Mat rf = imread(tmp); 
        draw(rf, right);
        for(int j=1;j<=leftn;j++)
        {
            sprintf(tmp, leftF, j);
            Mat lf = imread(tmp);
            if(checkface(rf, lf))
                draw(lf, left);
        }
        Mat result = mergeImg(left, right);
        imshow("result", result);
        waitKey();
    }
}

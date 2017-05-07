#include "opencv2/opencv.hpp"  
#include "opencv2/objdetect/objdetect.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
  
#include <iostream>  
#include <stdio.h>  
  
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
//    printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) ); 
    IplImage iplimg = img;
    IplImage *tmp = cvCloneImage(&iplimg); 
    for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )  
    {  
        IplImage* temp = cutImage(tmp, cvRect(r->x, r->y, r->width, r->height));  
        return temp;  
    }  
  
    return tmp;  
}  
//画直方图用  
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
  
    // CV_COMP_CHISQR,CV_COMP_BHATTACHARYYA这两种都可以用来做直方图的比较，值越小，说明图形越相似  
//    printf("CV_COMP_CHISQR : %.4f\n", cvCompareHist(Histogram1, Histogram2, CV_COMP_CHISQR));  
//    printf("CV_COMP_BHATTACHARYYA : %.4f\n", cvCompareHist(Histogram1, Histogram2, CV_COMP_BHATTACHARYYA));  
  
  
    // CV_COMP_CORREL, CV_COMP_INTERSECT这两种直方图的比较，值越大，说明图形越相似  
    fprintf(stderr, "CV_COMP_CORREL : %.4f\n", cvCompareHist(Histogram1, Histogram2, CV_COMP_CORREL));  
    fprintf(stderr, "CV_COMP_INTERSECT : %.4f\n", cvCompareHist(Histogram1, Histogram2, CV_COMP_INTERSECT)); 

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
  
    // CV_COMP_CHISQR,CV_COMP_BHATTACHARYYA这两种都可以用来做直方图的比较，值越小，说明图形越相似  
//    printf("CV_COMP_CHISQR : %.4f\n", cvCompareHist(Histogram1, Histogram2, CV_COMP_CHISQR));  
//    printf("CV_COMP_BHATTACHARYYA : %.4f\n", cvCompareHist(Histogram1, Histogram2, CV_COMP_BHATTACHARYYA));  
  
  
    // CV_COMP_CORREL, CV_COMP_INTERSECT这两种直方图的比较，值越大，说明图形越相似  
    fprintf(stderr, "CV_COMP_CORREL : %.4f\n", cvCompareHist(Histogram1, Histogram2, CV_COMP_CORREL));  
    fprintf(stderr, "CV_COMP_INTERSECT : %.4f\n", cvCompareHist(Histogram1, Histogram2, CV_COMP_INTERSECT)); 

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
int main(int argc, char* argv[])  
{  
	String srcImage = argv[1];  
	String targetImage = argv[2];  
    CascadeClassifier cascade;  
    if( !cascade.load( cascadeName ) )  
    {  
        return -1;  
    }  
  
    Mat srcImg, targetImg;  
    IplImage* faceImage1;  
    IplImage* faceImage2;  
    srcImg = imread(srcImage);  
    targetImg = imread(targetImage); 
    if(srcImg.empty() || targetImg.empty()) return 0; 
    faceImage1 = detect(srcImg, cascade, 1);  
    if (faceImage1 == NULL) {  
        return -1;  
    }  
//    cvSaveImage("d:\\face.jpg", faceImage1, 0);  
    faceImage2 = detect(targetImg, cascade, 1);  
    if (faceImage2 == NULL) {  
        return -1;  
    }  
//    cvSaveImage("d:\\face1.jpg", faceImage2, 0);  
    if(CompareLuv(faceImage1, faceImage2, 0) < 1.41 || CompareLuv(faceImage1, faceImage2, 2) < 1.43 || CompareHist(faceImage1, faceImage2) < 1.454) return 0; 
    
    printf("%s %s %lf %lf %lf\n", argv[1], argv[2], CompareLuv(faceImage1, faceImage2, 0), CompareLuv(faceImage1, faceImage2, 2), CompareHist(faceImage1, faceImage2)); 
    
//    namedWindow("image1");  
//    namedWindow("image2");  
//    imshow("image1", Mat(faceImage1));  
//    imshow("image2", Mat(faceImage2));  
//    waitKey();

    cvReleaseImage(&faceImage1);  
    cvReleaseImage(&faceImage2);  
    return 0;  
}  

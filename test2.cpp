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
    //imshow("original image1",img01);
    //imshow("original image2",img02);

    //SIFT特征检测
    SurfFeatureDetector detector;        //定义特点点检测器
    vector<KeyPoint> keypoint01,keypoint02;//定义两个容器存放特征点
    detector.detect(img01,keypoint01);
    detector.detect(img02,keypoint02);

    //在两幅图中画出检测到的特征点
    Mat out_img01;
    Mat out_img02;
    drawKeypoints(img01,keypoint01,out_img01);
    drawKeypoints(img02,keypoint02,out_img02);
    //imshow("特征点图01",out_img01);
    //imshow("特征点图02",out_img02);

    //提取特征点的特征向量（128维）
    SiftDescriptorExtractor extractor;
    Mat descriptor01,descriptor02;
    extractor.compute(img01,keypoint01,descriptor01);
    extractor.compute(img02,keypoint02,descriptor02);

    //匹配特征点，主要计算两个特征点特征向量的欧式距离，距离小于某个阈值则认为匹配

    BruteForceMatcher<L2<float> > matcher;
    vector<DMatch> matches;
    Mat img_matches;
    matcher.match(descriptor01,descriptor02,matches);
    drawMatches(img01,keypoint01,img02,keypoint02,matches,img_matches);
    imshow("误匹配消除前",img_matches);
    
    //RANSAC 消除误匹配特征点 主要分为三个部分：
    //1）根据matches将特征点对齐,将坐标转换为float类型
    //2）使用求基础矩阵方法 findFundamentalMat,得到RansacStatus
    //3）根据RansacStatus来将误匹配的点也即RansacStatus[i]=0的点删除

    //根据matches将特征点对齐,将坐标转换为float类型
    vector<KeyPoint> R_keypoint01,R_keypoint02;
    for (size_t i=0;i<matches.size();i++)   
    {
        R_keypoint01.push_back(keypoint01[matches[i].queryIdx]);
        R_keypoint02.push_back(keypoint02[matches[i].trainIdx]);
        //这两句话的理解：R_keypoint1是要存储img01中能与img02匹配的特征点，
        //matches中存储了这些匹配点对的img01和img02的索引值
    }

    //坐标转换
    vector<Point2f>p01,p02;
    for (size_t i=0;i<matches.size();i++)
    {
        p01.push_back(R_keypoint01[i].pt);
        p02.push_back(R_keypoint02[i].pt);
    }

    //利用基础矩阵剔除误匹配点
    vector<uchar> RansacStatus;
    Mat Fundamental= findFundamentalMat(p01,p02,RansacStatus,FM_RANSAC);


    vector<KeyPoint> RR_keypoint01,RR_keypoint02;
    vector<DMatch> RR_matches;            //重新定义RR_keypoint 和RR_matches来存储新的关键点和匹配矩阵
    int index=0;
    for (size_t i=0;i<matches.size();i++)
    {
        if (RansacStatus[i]!=0)
        {
            RR_keypoint01.push_back(R_keypoint01[i]);
            RR_keypoint02.push_back(R_keypoint02[i]);
            matches[i].queryIdx=index;
            matches[i].trainIdx=index;
            RR_matches.push_back(matches[i]);
            index++;
        }
    }
    Mat img_RR_matches;
    drawMatches(img01,RR_keypoint01,img02,RR_keypoint02,RR_matches,img_RR_matches);
    imshow("消除误匹配点后",img_RR_matches);
    printf("%lu %lu\n", matches.size(), RR_matches.size());
    waitKey();
}

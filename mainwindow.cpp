#include "mainwindow.h"
#include <QScrollArea>
#include <QBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QSlider>
#include <cmath>
#include <QIntValidator>
#include <vector>
#include <cstring>

MainWindow::MainWindow(QWidget *parent) :
    QWidget(parent)
{
    this->setWindowTitle("Digital Image Processing Experiment 1");
    this->resize(600, 350);
    scrollarea = new QScrollArea(this);
    scrollarea->setMinimumSize(300, 300);
    imgLabel = new QLabel(this);
    scrollarea->setWidget(imgLabel);
    scrollarea->setSizePolicy(QSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding));
    scrollarea->setAlignment(Qt::AlignCenter);
    scrollarea->setWidgetResizable(true);
    QHBoxLayout *hbox = new QHBoxLayout(this);
    QWidget *widget = new QWidget(this);
    widget->setMaximumWidth(600);
    QVBoxLayout *vbox = new QVBoxLayout(widget);
    hbox->addWidget(scrollarea);
    hbox->addWidget(widget);
    this->setLayout(hbox);

    QPushButton *openButton = new QPushButton("Open");
    QPushButton *saveButton = new QPushButton("Save");
    QHBoxLayout *htbox = new QHBoxLayout();
    htbox->addWidget(openButton);
    htbox->addWidget(saveButton);
    vbox->addLayout(htbox);
    connect(openButton, SIGNAL(clicked(bool)), this, SLOT(openImg()));
    connect(saveButton, SIGNAL(clicked(bool)), this, SLOT(saveImg()));

    brightSlider = new QSlider(Qt::Horizontal);
    QLabel *brightLabel = new QLabel("Brightness\t");
    brightSlider->setMinimum(-256);
    brightSlider->setMaximum(256);
    brightSlider->setValue(0);
    connect(brightSlider, SIGNAL(valueChanged(int)), this, SLOT(changeBright(int)));
    htbox = new QHBoxLayout();
    htbox->addWidget(brightLabel);
    htbox->addWidget(brightSlider);
    vbox->addLayout(htbox);

    contrastSlider = new QSlider(Qt::Horizontal);
    QLabel *contrastLabel = new QLabel("Contrast\t\t");
    contrastSlider->setMinimum(-256);
    contrastSlider->setMaximum(256);
    contrastSlider->setValue(0);
    connect(contrastSlider, SIGNAL(valueChanged(int)), this, SLOT(changeContrast(int)));
    htbox = new QHBoxLayout();
    htbox->addWidget(contrastLabel);
    htbox->addWidget(contrastSlider);
    vbox->addLayout(htbox);

    gammaSlider = new QSlider(Qt::Horizontal);
    QLabel *gammaLabel = new QLabel("Gamma\t\t");
    gammaSlider->setMinimum(-256);
    gammaSlider->setMaximum(256);
    gammaSlider->setValue(0);
    connect(gammaSlider, SIGNAL(valueChanged(int)), this, SLOT(changeGamma(int)));
    htbox = new QHBoxLayout();
    htbox->addWidget(gammaLabel);
    htbox->addWidget(gammaSlider);
    vbox->addLayout(htbox);

    QPushButton *stretchButton = new QPushButton("Contrast Stretch");
    miEdit = new QLineEdit(this);
    miEdit->setValidator(new QIntValidator(0, 255, this));
    miEdit->setText("0");
    maEdit = new QLineEdit(this);
    maEdit->setText("255");
    maEdit->setValidator(new QIntValidator(0, 255, this));
    htbox = new QHBoxLayout();
    htbox->addWidget(stretchButton);
    htbox->addWidget(miEdit);
    htbox->addWidget(new QLabel("~", this));
    htbox->addWidget(maEdit);
    vbox->addLayout(htbox);
    connect(stretchButton, SIGNAL(clicked(bool)), this, SLOT(contrastStretch()));

    QPushButton *histEqualButton = new QPushButton("Hist Equalization (Luma)");
    QPushButton *histEqualRGBButton = new QPushButton("Hist Equalization (RGB)");
    htbox = new QHBoxLayout();
    htbox->addWidget(histEqualButton);
    htbox->addWidget(histEqualRGBButton);
    vbox->addLayout(htbox);
    connect(histEqualButton, SIGNAL(clicked(bool)), this, SLOT(histEqual()));
    connect(histEqualRGBButton, SIGNAL(clicked(bool)), this, SLOT(histEqualRGB()));

    QPushButton *histMatchButton = new QPushButton("Hist Matching (Luma)");
    QPushButton *histMatchRGBButton = new QPushButton("Hist Matching (RGB)");
    htbox = new QHBoxLayout();
    htbox->addWidget(histMatchButton);
    htbox->addWidget(histMatchRGBButton);
    vbox->addLayout(htbox);
    connect(histMatchButton, SIGNAL(clicked(bool)), this, SLOT(histMatch()));
    connect(histMatchRGBButton, SIGNAL(clicked(bool)), this, SLOT(histMatchRGB()));

    heightEdit = new QLineEdit(this);
    heightEdit->setValidator(new QIntValidator(0, 1024*1024, this));
    heightEdit->setText("1024");
    weightEdit = new QLineEdit(this);
    weightEdit->setValidator(new QIntValidator(0, 1024*1024, this));
    weightEdit->setText("1024");
    htbox = new QHBoxLayout();
    htbox->addWidget(new QLabel("New Size\t", this));
    htbox->addWidget(weightEdit);
    htbox->addWidget(new QLabel("X", this));
    htbox->addWidget(heightEdit);
    vbox->addLayout(htbox);

    QPushButton *neighborButton = new QPushButton("Nearest neighbor");
    QPushButton *bilinearButton = new QPushButton("Bilinear");
    QPushButton *bicubicButton = new QPushButton("Bicubic");
    htbox = new QHBoxLayout();
    htbox->addWidget(neighborButton);
    htbox->addWidget(bilinearButton);
    htbox->addWidget(bicubicButton);
    vbox->addLayout(htbox);
    connect(neighborButton, SIGNAL(clicked(bool)), this, SLOT(neighborResize()));
    connect(bilinearButton, SIGNAL(clicked(bool)), this, SLOT(bilinearResize()));
    connect(bicubicButton, SIGNAL(clicked(bool)), this, SLOT(bicubicResize()));

}

void MainWindow::displayImg()
{
    QImage qimg = QImage((const unsigned char*)(img.data), img.cols, img.rows, img.cols*img.channels(), QImage::Format_RGB888);
    this->imgLabel->setPixmap(QPixmap::fromImage(qimg));
    this->imgLabel->setAlignment(Qt::AlignCenter);
}

void MainWindow::openImg()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open Image"), "",tr("Image File(*.png *.jpg *.jpeg *.bmp);;All File(*.*)"));
    if (fileName.isEmpty()) return;
    this->img = cv::imread(fileName.toStdString());
    cv::cvtColor(this->img, this->img, cv::COLOR_BGR2RGB);
    this->state = -1;
    this->brightSlider->setValue(0);
    this->contrastSlider->setValue(0);
    this->gammaSlider->setValue(0);
    displayImg();
}

void MainWindow::saveImg()
{
    if(img.empty()) return ;
    QString fileName = QFileDialog::getSaveFileName(this, tr("Save Image"), "",tr("Image File(*.png *.jpg *.jpeg *.bmp);;All File(*.*)"));
    if (fileName.isEmpty()) return;
    cv::Mat timg;
    cv::cvtColor(this->img, timg, cv::COLOR_RGB2BGR);
    cv::imwrite(fileName.toStdString(), timg);
}

static void myLUT(cv::Mat& src, const uchar* table, cv::Mat& trg)
{
    trg = src.clone();
    // accept only char type matrices
    CV_Assert(trg.depth() != sizeof(uchar));
    int channels = trg.channels();
    int nRows = trg.rows ;
    int nCols = trg.cols* channels;
    if (trg.isContinuous())
    {
        nCols *= nRows;
        nRows = 1;
    }
    for(int i = 0; i < nRows; ++i)
    {
        uchar* p = trg.ptr<uchar>(i);
        for (int j = 0; j < nCols; ++j)
        {
            p[j] = table[p[j]];
        }
    }
    return ;
}

static void minmaxImgage(const cv::Mat& src, uchar &mi, uchar &ma)
{
    CV_Assert(src.depth() != sizeof(uchar));
    int channels = src.channels();
    int nRows = src.rows ;
    int nCols = src.cols* channels;
    if (src.isContinuous())
    {
        nCols *= nRows;
        nRows = 1;
    }
    mi = 255, ma = 0;
    for(int i = 0; i < nRows; ++i)
    {
        const uchar* p = src.ptr<uchar>(i);
        for (int j = 0; j < nCols; ++j)
        {
            mi = p[j] < mi ? p[j] : mi;
            ma = p[j] > ma ? p[j] : ma;
        }
    }
    return ;
}

static std::vector<float> getCdf(const cv::Mat& src)
{
    CV_Assert(src.depth() != sizeof(uchar));
    int channels = src.channels();
    int nRows = src.rows ;
    int nCols = src.cols* channels;
    if (src.isContinuous())
    {
        nCols *= nRows;
        nRows = 1;
    }
    int num[256];
    std::vector<float> ans;
    memset(num, 0, sizeof(num));
    for(int i = 0; i < nRows; ++i)
    {
        const uchar* p = src.ptr<uchar>(i);
        for (int j = 0; j < nCols; ++j)
        {
            num[p[j]]++;
        }
    }
    for(int i=1;i<256;i++) num[i]+=num[i-1];
    for(int i=0;i<256;i++) ans.push_back(double(num[i])/(nRows*nCols));
    return ans;
}


void MainWindow::changeBright(int delta)
{
    if(img.empty()) return ;
    if(state != 0)
    {
        this->imgbak = this->img.clone();
        state = 0;
    }
    uchar p[256];
    for (int i = 0; i < 256; i++)
    {
        int t = i+delta;
        if(t > 255) t=255;
        if(t < 0) t=0;
        p[i] = (uchar)t;
    }
    myLUT(this->imgbak, p, this->img);
    this->displayImg();
}

void MainWindow::changeContrast(int delta)
{
    if(img.empty()) return ;
    if(state != 1)
    {
        this->imgbak = this->img.clone();
        state = 1;
    }
    uchar p[256];
    float g = float(delta+256)/256;
    for (int i = 0; i < 256; i++)
    {
        int t = int((i - 127) * g + 127);
        if(t > 255) t=255;
        if(t < 0) t=0;
        p[i] = (uchar)t;
    }
    myLUT(this->imgbak, p, this->img);
    this->displayImg();
}

void MainWindow::changeGamma(int delta)
{
    if(img.empty()) return ;
    if(state != 2)
    {
        this->imgbak = this->img.clone();
        state = 2;
    }
    uchar p[256];
    float g = 1.0 / (float(delta+256)/256);
    for (int i = 0; i < 256; i++)
    {
        int t = int(255 * pow(float(i) / 255,  g));
        if(t > 255) t=255;
        if(t < 0) t=0;
        p[i] = (uchar)t;
    }
    myLUT(this->imgbak, p, this->img);
    this->displayImg();
}

void MainWindow::contrastStretch()
{
    if(img.empty()) return ;
    if(miEdit->text().toInt() > maEdit->text().toInt()) return ;
    if(state != 3)
    {
        this->imgbak = this->img.clone();
        state = 3;
    }
    uchar m = miEdit->text().toUInt(), M = maEdit->text().toUInt(), mi, Mi;
    minmaxImgage(this->imgbak, mi, Mi);
    uchar p[256];
    for (int i = 0; i < 256; i++)
    {
        int t = int((M - m) * (i - mi) / float(Mi - mi) + m);
        if(t > 255) t=255;
        if(t < 0) t=0;
        p[i] = (uchar)t;
    }
    myLUT(this->imgbak, p, this->img);
    this->displayImg();
}

void MainWindow::histEqual()
{
    if(img.empty()) return ;
    if(state != 4)
    {
        this->imgbak = this->img.clone();
        state = 4;
    }
    cv::Mat yimg;
    cv::cvtColor(imgbak, yimg, cv::COLOR_RGB2YCrCb);
    std::vector<cv::Mat> ycr;
    cv::split(yimg, ycr);
    std::vector<float> cdf = getCdf(ycr[0]);
    uchar p[256];
    for (int i = 0; i < 256; i++)
    {
        int t = cdf[i]*255;
        if(t > 255) t=255;
        if(t < 0) t=0;
        p[i] = (uchar)t;
    }
    myLUT(ycr[0], p, ycr[0]);
    cv::merge(ycr, yimg);
    cv::cvtColor(yimg, img, cv::COLOR_YCrCb2RGB);
    this->displayImg();
}

void MainWindow::histEqualRGB()
{
    if(img.empty()) return ;
    if(state != 5)
    {
        this->imgbak = this->img.clone();
        state = 5;
    }
    std::vector<cv::Mat> ycr;
    cv::split(imgbak, ycr);
    for(int c=0;c<imgbak.channels();c++)
    {
        std::vector<float> cdf = getCdf(ycr[c]);
        uchar p[256];
        for (int i = 0; i < 256; i++)
        {
            int t = cdf[i]*255;
            if(t > 255) t=255;
            if(t < 0) t=0;
            p[i] = (uchar)t;
        }
        myLUT(ycr[c], p, ycr[c]);
    }
    cv::merge(ycr, img);
    this->displayImg();
}

void MainWindow::histMatch()
{
    if(img.empty()) return ;
    QString fileName = QFileDialog::getOpenFileName(this, tr("Target Image"), "",tr("Image File(*.png *.jpg *.jpeg *.bmp);;All File(*.*)"));
    if (fileName.isEmpty()) return;
    cv::Mat trg = cv::imread(fileName.toStdString());
    if(state != 6)
    {
        this->imgbak = this->img.clone();
        state = 6;
    }

    cv::Mat yimg, yimg_trg;
    cv::cvtColor(imgbak, yimg, cv::COLOR_RGB2YCrCb);
    cv::cvtColor(trg, yimg_trg, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> ycr, ycr_trg;
    cv::split(yimg, ycr);
    cv::split(yimg_trg, ycr_trg);
    std::vector<float> cdf = getCdf(ycr[0]);
    std::vector<float> cdf_trg = getCdf(ycr_trg[0]);
    uchar p[256];
    for (int i = 0, j = 0; i < 256; i++)
    {
        while(cdf_trg[j] < cdf[i] && j < 255) j++;
        p[i] = j;
    }
    myLUT(ycr[0], p, ycr[0]);
    cv::merge(ycr, yimg);
    cv::cvtColor(yimg, img, cv::COLOR_YCrCb2RGB);
    this->displayImg();
}

void MainWindow::histMatchRGB()
{
    if(img.empty()) return ;
    QString fileName = QFileDialog::getOpenFileName(this, tr("Target Image"), "",tr("Image File(*.png *.jpg *.jpeg *.bmp);;All File(*.*)"));
    if (fileName.isEmpty()) return;
    cv::Mat trg = cv::imread(fileName.toStdString());
    if(state != 6)
    {
        this->imgbak = this->img.clone();
        state = 6;
    }

    cv::cvtColor(trg, trg, cv::COLOR_BGR2RGB);
    std::vector<cv::Mat> ycr, ycr_trg;
    cv::split(imgbak, ycr);
    cv::split(trg, ycr_trg);
    for(int c=0; c<imgbak.channels(); c++)
    {
        std::vector<float> cdf = getCdf(ycr[c]);
        std::vector<float> cdf_trg = getCdf(ycr_trg[c]);
        uchar p[256];
        for (int i = 0, j = 0; i < 256; i++)
        {
            while(cdf_trg[j] < cdf[i] && j < 255) j++;
            p[i] = j;
        }
        myLUT(ycr[c], p, ycr[c]);
    }
    cv::merge(ycr, img);
    this->displayImg();
}

static double getPSNR(const cv::Mat& I1, const cv::Mat& I2)
{
    cv::Mat s1;
    cv::absdiff(I1, I2, s1);
    s1.convertTo(s1, CV_32F);
    s1 = s1.mul(s1);

    cv::Scalar s = cv::sum(s1);

    double sse = s.val[0] + s.val[1] + s.val[2];

//    if(sse <= 1e-10)
//        return 0;
//    else
    {
        double mse = sse/(double)(I1.channels()*I1.total());
        double psnr = 10.0*log10((255*255)/mse);
        return psnr;
    }
}

void MainWindow::neighbor(const cv::Mat &imgbak, cv::Mat &img, int h, int w)
{
    img.create(h,w, CV_8UC3);
    int nRows = img.rows;
    int nCols = img.cols;
    for(int i = 0; i < nRows; ++i)
    {
        cv::Vec3b* n = img.ptr<cv::Vec3b>(i);
        const cv::Vec3b* p = imgbak.ptr<cv::Vec3b>(round(double(i)/nRows*imgbak.rows));
        for (int j = 0; j < nCols; ++j)
        {
            n[j] = p[int(round(double(j)/nCols*imgbak.cols))];
        }
    }
}

void MainWindow::neighborResize()
{
    if(img.empty()) return ;
    if(heightEdit->text().toInt() * weightEdit->text().toInt() == 0) return ;
    if(state != 7)
    {
        this->imgbak = this->img.clone();
        state = 7;
    }
    cv::Mat tmp;
    neighbor(imgbak, img, heightEdit->text().toInt(), weightEdit->text().toInt());
    neighbor(img, tmp, imgbak.rows, imgbak.cols);
    QMessageBox::information(this, "PSNR", QString::number(getPSNR(tmp, imgbak)));
    this->displayImg();
}

static cv::Vec3b bilinearInterpolate(cv::Vec3b c00, cv::Vec3b c10, cv::Vec3b c01, cv::Vec3b c11, double w1, double w2, double w3, double w4)
{
    return w1*c00+w2*c01+w3*c10+w4*c11;
}

void MainWindow::bilinear(const cv::Mat &imgbak, cv::Mat &img, int h, int w)
{
    img.create(h,w, CV_8UC3);
    int nRows = img.rows;
    int nCols = img.cols;
    for(int i = 0; i < nRows; ++i)
    {
        cv::Vec3b* n = img.ptr<cv::Vec3b>(i);
        int x = double(i)/nRows*imgbak.rows;
        const cv::Vec3b* p0 = imgbak.ptr<cv::Vec3b>(x);
        const cv::Vec3b* p1 = imgbak.ptr<cv::Vec3b>(x+1);
        double xDiff = double(i)/nRows*imgbak.rows-x;
        double xDiffR = 1.0f-xDiff;
        #pragma omp parallel for
        for (int j = 0; j < nCols; ++j)
        {
            int y = double(j)/nCols*imgbak.cols;
            double yDiff = double(j)/nCols*imgbak.cols-y;
            double yDiffR = 1.0f-yDiff;
            const cv::Vec3b c00=p0[y], c01=p0[y+1], c10=p1[y], c11=p1[y+1];
            double w1 = yDiffR*xDiffR;
            double w2 = yDiff*xDiffR;
            double w3 = yDiffR*xDiff;
            double w4 = yDiff*xDiff;
            n[j] = bilinearInterpolate(c00, c10, c01, c11, w1, w2, w3, w4);
        }
    }
}

void MainWindow::bilinearResize()
{
    if(img.empty()) return ;
    if(heightEdit->text().toInt() * weightEdit->text().toInt() == 0) return ;
    if(state != 7)
    {
        this->imgbak = this->img.clone();
        state = 7;
    }
    cv::Mat tmp;
    bilinear(imgbak, img, heightEdit->text().toInt(), weightEdit->text().toInt());
    bilinear(img, tmp, imgbak.rows, imgbak.cols);
    QMessageBox::information(this, "PSNR", QString::number(getPSNR(tmp, imgbak)));
    this->displayImg();
}

static double cubicW(double x)
{
    const double a = -0.5;
    if(x < 1+1e-10) return (a+2)*x*x*x-(a+3)*x*x+1;
    if(x < 2) return a*x*x*x-5*a*x*x+8*a*x-4*a;
    return 0;
}

void MainWindow::bicubic(const cv::Mat &imgbak, cv::Mat &img, int h, int w)
{
    img.create(h,w, CV_8UC3);
    int nRows = img.rows;
    int nCols = img.cols;
    for(int i = 0; i < nRows; ++i)
    {
        cv::Vec3b* n = img.ptr<cv::Vec3b>(i);
        int x = double(i)/nRows*imgbak.rows;
        const cv::Vec3b* pt[4];
        const cv::Vec3b* *p=pt+1;
        p[0] = imgbak.ptr<cv::Vec3b>(x);
        for(int k=-1;k<=2;k++)
        {
            if(x+k < 0) p[k] = p[k+1];
            else if(x+k >= imgbak.rows) p[k] = p[k-1];
            else p[k] = imgbak.ptr<cv::Vec3b>(x+k);
        }
        #pragma omp parallel for
        for (int j = 0; j < nCols; ++j)
        {
            int y = double(j)/nCols*imgbak.cols;
            cv::Vec3f sum = cv::Vec3f();
            double sumt = 0.0;
            for (int k=-1; k<=2;k++)
                for (int l=-1; l<=2; l++)
                {
                    cv::Vec3f t;
                    if(y+l < 0) t = p[k][0];
                    else if(y+l >= imgbak.cols) t = p[k][imgbak.cols];
                    else t = p[k][y+l];
                    double w = cubicW(fabs(double(j)/nCols*imgbak.cols-(y+l)))*cubicW(fabs(double(i)/nRows*imgbak.rows-(x+k)));
                    sum = sum + w * t;
                    sumt += w;
                }
            n[j] = sum/sumt;
        }
    }
}

void MainWindow::bicubicResize()
{
    if(img.empty()) return ;
    if(heightEdit->text().toInt() * weightEdit->text().toInt() == 0) return ;
    if(state != 7)
    {
        this->imgbak = this->img.clone();
        state = 7;
    }
    cv::Mat tmp;
    bicubic(imgbak, img, heightEdit->text().toInt(), weightEdit->text().toInt());
    bicubic(img, tmp, imgbak.rows, imgbak.cols);
    QMessageBox::information(this, "PSNR", QString::number(getPSNR(tmp, imgbak)));
    this->displayImg();
}


MainWindow::~MainWindow()
{
}

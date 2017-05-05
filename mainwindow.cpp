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
    QHBoxLayout *hbox = new QHBoxLayout(this);
    QVBoxLayout *vbox = new QVBoxLayout();
    hbox->addWidget(scrollarea);
    hbox->addLayout(vbox);
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
    htbox->addWidget(new QLabel(" to ", this));
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
}

void MainWindow::displayImg()
{
    QImage qimg = QImage((const unsigned char*)(img.data), img.cols, img.rows, img.cols*img.channels(), QImage::Format_RGB888);
    this->imgLabel->setPixmap(QPixmap::fromImage(qimg));
    this->imgLabel->resize(this->imgLabel->pixmap()->size());
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

void myLUT(cv::Mat& src, const uchar* table, cv::Mat& trg)
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

void minmaxImgage(const cv::Mat& src, uchar &mi, uchar &ma)
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

std::vector<float> getCdf(const cv::Mat& src)
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
    for(int c=0;c<3;c++)
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
    for(int c=0; c<3; c++)
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

MainWindow::~MainWindow()
{
}

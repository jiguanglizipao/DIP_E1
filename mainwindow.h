#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/opencv.hpp>
#include <QLabel>
#include <QPushButton>
#include <QWidget>
#include <QScrollArea>
#include <QSlider>
#include <QLineEdit>

class MainWindow : public QWidget
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    cv::Mat img, imgbak;
    QScrollArea *scrollarea;
    QSlider *brightSlider, *contrastSlider, *gammaSlider;
    QLineEdit *miEdit, *maEdit, *heightEdit, *weightEdit;
    QLabel *imgLabel;
    QImage qimg;
    int state;
    void displayImg();
private slots:
    void openImg();
    void saveImg();
    void changeBright(int delta);
    void changeContrast(int delta);
    void changeGamma(int delta);
    void contrastStretch();
    void histEqual();
    void histEqualRGB();
    void histMatch();
    void histMatchRGB();
    void neighborResize();
    void bilinearResize();
    void bicubicResize();
    void neighbor(const cv::Mat &imgbak, cv::Mat &img, int h, int w);
    void bilinear(const cv::Mat &imgbak, cv::Mat &img, int h, int w);
    void bicubic(const cv::Mat &imgbak, cv::Mat &img, int h, int w);
};

#endif // MAINWINDOW_H

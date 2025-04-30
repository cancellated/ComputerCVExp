#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <opencv2/opencv.hpp>
#include <qlabel.h>
#include <QMessageBox>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_btnHalfInverse_clicked()
{
    currentFunction = "halfInverse";
    ui->stackedWidget->setCurrentWidget(ui->functionPage);
}


void MainWindow::on_btnEdgeDetect_clicked()
{
    currentFunction = "edgeDetect";
    ui->stackedWidget->setCurrentWidget(ui->functionPage);
}


void MainWindow::on_btnBack_clicked()
{
    // 清除显示的图片
    ui->lblImageDisplay_1->clear();
    ui->lblImageDisplay_2->clear();
    
    // 清空图像数据
    originalImage.release();
    processedImage.release();
    
    // 切换页面
    ui->stackedWidget->setCurrentWidget(ui->menuPage);
}


void MainWindow::on_btnSelectFile_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, "选择图片", "", "图片文件 (*.jpg *.png *.bmp)");
    if(fileName.isEmpty()) {
        qDebug() << "用户取消了文件选择";
        return;
    }

    // 检查文件是否存在
    QFile file(fileName);
    if(!file.exists()) {
        QMessageBox::critical(this, "错误", QString("文件不存在: %1").arg(fileName));
        return;
    }

    // 尝试读取文件
    try {
        QByteArray ba = fileName.toLocal8Bit();
        originalImage = cv::imread(ba.constData(), cv::IMREAD_UNCHANGED);
        if(originalImage.empty()) {
            QMessageBox::critical(this, "错误", "无法解码图片");
            return;
        }
    }
    catch(...) {
        QMessageBox::critical(this, "错误", "发生未知错误");
        return;
    }

    // 使用临时副本转换颜色供显示
    cv::Mat displayOriginal;
    if(originalImage.channels() == 4) {
        cv::cvtColor(originalImage, displayOriginal, cv::COLOR_BGRA2RGBA);
    }
    else if(originalImage.channels() == 3) {
        cv::cvtColor(originalImage, displayOriginal, cv::COLOR_BGR2RGB);
    }
    else {
        displayOriginal = originalImage;
    }

    QImage originalImg;
    switch(displayOriginal.channels()) {
    case 4:
        originalImg = QImage(displayOriginal.data, displayOriginal.cols, displayOriginal.rows,
                           displayOriginal.step, QImage::Format_RGBA8888);
        break;
    case 3:
        originalImg = QImage(displayOriginal.data, displayOriginal.cols, displayOriginal.rows,
                           displayOriginal.step, QImage::Format_RGB888);
        break;
    default:
        originalImg = QImage(displayOriginal.data, displayOriginal.cols, displayOriginal.rows,
                           displayOriginal.step, QImage::Format_Grayscale8);
    }

    ui->lblImageDisplay_1->setPixmap(QPixmap::fromImage(originalImg).scaled(
        ui->lblImageDisplay_1->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));

    processImage();
}

void MainWindow::processImage()
{
    if(originalImage.empty()) return;

    static const std::unordered_map<std::string, std::function<void(cv::Mat&, cv::Mat&)>> processors = {
        {"halfInverse", [](cv::Mat& src, cv::Mat& dst) {
            dst = src.clone();
            int half_height = dst.rows / 2;

            // 强制统一为BGR格式处理
            if(dst.channels() == 1) {
                cv::cvtColor(dst, dst, cv::COLOR_GRAY2BGR);
            }
            else if(dst.channels() == 4) {
                cv::cvtColor(dst, dst, cv::COLOR_BGRA2BGR);
            }

            cv::Mat top_half = dst.rowRange(0, half_height);
            cv::bitwise_not(top_half, top_half);

            // 直接复制原图下半部分
            cv::Mat bottom_half = dst.rowRange(half_height, dst.rows);
            src.rowRange(half_height, src.rows).copyTo(bottom_half);
        }},
        {"edgeDetect", [](cv::Mat& src, cv::Mat& dst) {
            cv::Mat gray, edges;
            cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
            cv::Canny(gray, edges, 50, 150);
            cv::cvtColor(edges, dst, cv::COLOR_GRAY2BGR);
        }}
    };

    // 将QString转换为std::string
    std::string funcName = currentFunction.toStdString();
    
    // 查找并执行处理函数
    auto it = processors.find(funcName);
    if(it != processors.end()) {
        it->second(originalImage, processedImage);
    }

    // 处理后的颜色转换
    cv::Mat displayProcessed;
    if(processedImage.channels() == 4) {
        cv::cvtColor(processedImage, displayProcessed, cv::COLOR_BGRA2RGBA);
    }
    else if(processedImage.channels() == 3) {
        cv::cvtColor(processedImage, displayProcessed, cv::COLOR_BGR2RGB);
    }
    else {
        displayProcessed = processedImage;
    }

    QImage processedImg;
    switch(displayProcessed.type()) {
    case CV_8UC1:
        processedImg = QImage(displayProcessed.data, displayProcessed.cols, displayProcessed.rows,
                            displayProcessed.step, QImage::Format_Grayscale8);
        break;
    case CV_8UC3:
        processedImg = QImage(displayProcessed.data, displayProcessed.cols, displayProcessed.rows,
                            displayProcessed.step, QImage::Format_RGB888);
        break;
    case CV_8UC4:
        processedImg = QImage(displayProcessed.data, displayProcessed.cols, displayProcessed.rows,
                            displayProcessed.step, QImage::Format_RGBA8888);
        break;
    }

    ui->lblImageDisplay_2->setPixmap(QPixmap::fromImage(processedImg).scaled(
        ui->lblImageDisplay_2->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}


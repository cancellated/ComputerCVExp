#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <opencv2/opencv.hpp>
#include <qlabel.h>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QPen>
#include <QPainterPath>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->lblImageDisplay_1->installEventFilter(this);
    ui->lblImageDisplay_1->setMouseTracking(true);
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

void MainWindow::on_btnROI_clicked()
{
    currentFunction = "roi";
    ui->stackedWidget->setCurrentWidget(ui->functionPage);
}

void MainWindow::on_btnShapeAnalyze_clicked()
{
    currentFunction = "shapeAnalyze";
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
    ui->lblParameters->setVisible(false);
    ui->stackedWidget->setCurrentWidget(ui->menuPage);
}


void MainWindow::on_btnSelectFile_clicked()
{


    QFileDialog dialog(this, "选择图片", "", "图片文件 (*.jpg *.png *.bmp)");
    if(dialog.exec() == QDialog::Accepted) {
        QString fileName = dialog.selectedFiles().first();
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

        originalPixmap = QPixmap::fromImage(originalImg);

        // 如果是ROI模式，进入选择状态
        if(currentFunction == "roi" || currentFunction == "shapeAnalyze") {
            this->lassoPoints.clear();
            this->roiMask.release();
            ui->lblImageDisplay_2->clear();
            isSelectingROI = true;
            ui->lblImageDisplay_1->setCursor(Qt::CrossCursor);
        } else {
            processImage(); // 其他模式
        }

    }
}

void MainWindow::processImage()
{
    if(originalImage.empty()) return;

    // 默认隐藏参数label
    ui->lblParameters->setVisible(false);

    static const std::unordered_map<std::string, std::function<void(cv::Mat&, cv::Mat&)>> processors = {
        {"halfInverse", [](cv::Mat& src, cv::Mat& dst) {
            dst = src.clone();
            int half_height = dst.rows / 2;

            // 统一为BGR格式处理
            if(dst.channels() == 1) {
                cv::cvtColor(dst, dst, cv::COLOR_GRAY2BGR);
            }
            else if(dst.channels() == 4) {
                cv::cvtColor(dst, dst, cv::COLOR_BGRA2BGR);
            }

            cv::Mat top_half = dst.rowRange(0, half_height);
            cv::bitwise_not(top_half, top_half);

            // 复制原图下半部分
            cv::Mat bottom_half = dst.rowRange(half_height, dst.rows);
            src.rowRange(half_height, src.rows).copyTo(bottom_half);
        }},
        {"edgeDetect", [](cv::Mat& src, cv::Mat& dst) {
            // 转换为灰度图
            cv::Mat gray;
            cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
            
            // SUSAN算子
            cv::Mat susanEdges = cv::Mat::zeros(gray.size(), CV_8UC1);
            cv::Mat corners = cv::Mat::zeros(gray.size(), CV_8UC1);
            
            // SUSAN参数
            int radius = 3;
            
            // 实现SUSAN算法
            for(int y = radius; y < gray.rows - radius; y++) {
                for(int x = radius; x < gray.cols - radius; x++) {
                    int similar = 0;
                    std::vector<cv::Point> cornerPoints;
                    
                    // 检查圆形邻域
                    for(int i = -radius; i <= radius; i++) {
                        for(int j = -radius; j <= radius; j++) {
                            if(i*i + j*j <= radius*radius) {
                                if(abs(gray.at<uchar>(y, x) - gray.at<uchar>(y+i, x+j)) < 25) {
                                    similar++;
                                    cornerPoints.emplace_back(x+j, y+i);
                                }
                            }
                        }
                    }
                    
                    // 边缘检测
                    if(similar < 25) {
                        susanEdges.at<uchar>(y, x) = 255;
                    }
                    
                    // 角点检测
                    if(similar <= 3) {
                        corners.at<uchar>(y, x) = 255;
                    }
                }
            }
            
            // 合并结果并标记角点
            cv::cvtColor(susanEdges, dst, cv::COLOR_GRAY2BGR);
            
            // 用红色标记角点
            for(int y = 0; y < corners.rows; y++) {
                for(int x = 0; x < corners.cols; x++) {
                    if(corners.at<uchar>(y, x) == 255) {
                        cv::circle(dst, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), -1);
                    }
                }
            }
        }},
        {"roi", [this](cv::Mat& src, cv::Mat& dst) {
            qDebug() << "处理前遮罩状态:" << roiMask.empty() << "尺寸:" << roiMask.cols << "x" << roiMask.rows;
            if(roiMask.empty() || roiMask.size() != src.size()) {
                qDebug() << "ROI处理: 遮罩为空或尺寸不匹配";
                qDebug() << "遮罩尺寸:" << roiMask.cols << "x" << roiMask.rows 
                         << "源图尺寸:" << src.cols << "x" << src.rows;
                dst = src.clone();
                return;
            }
            
            if(roiMask.type() != CV_8UC1) {
                cv::Mat tempMask;
                roiMask.convertTo(tempMask, CV_8UC1);
                src.copyTo(dst, tempMask);
            } else {
                src.copyTo(dst, roiMask);
            }
            
            // 计算ROI区域平均灰度值
            cv::Mat gray;
            if(src.channels() > 1) {
                cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
            } else {
                gray = src.clone();
            }
            cv::Scalar meanVal = cv::mean(gray, roiMask);
            
            // 参数显示
            cv::Rect boundingRect = cv::boundingRect(roiMask);
            qDebug() << "ROI区域参数: x=" << boundingRect.x << "y=" << boundingRect.y 
                     << "width=" << boundingRect.width << "height=" << boundingRect.height;
            
            regionParams = QString("ROI区域:\nX: %1\nY: %2\n宽度: %3\n高度: %4\n平均灰度值: %5")
                          .arg(boundingRect.x).arg(boundingRect.y)
                          .arg(boundingRect.width).arg(boundingRect.height)
                          .arg(meanVal[0], 0, 'f', 2);
            ui->lblParameters->setText(regionParams);
            ui->lblParameters->setAlignment(Qt::AlignCenter);
            ui->lblParameters->setWordWrap(true);
            ui->lblParameters->setVisible(true);
        }},
        {"shapeAnalyze", [this](cv::Mat& src, cv::Mat& dst) {
            // 检查是否有ROI遮罩
            if(roiMask.empty()) {
                dst = src.clone();
                return;
            }

            // 1. 转换为灰度图
            cv::Mat gray;
            if(src.channels() > 1) {
                cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
            } else {
                gray = src.clone();
            }

            // 2. 二值化
            cv::Mat binary;
            cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

            // 3. 应用ROI遮罩
            cv::Mat maskedBinary;
            binary.copyTo(maskedBinary, roiMask);

            // 4. 计算轮廓
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(maskedBinary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            // 创建彩色图像用于绘制
            cv::Mat drawImage;
            if(src.channels() == 1) {
                cv::cvtColor(src, drawImage, cv::COLOR_GRAY2BGR);
            } else {
                if(src.channels() == 4) {
                    cv::cvtColor(src, drawImage, cv::COLOR_BGRA2BGR);
                } else {
                    src.copyTo(drawImage);
                }
            }

            // 5. 计算各项特征并绘制
            for(auto& contour : contours) {
                // 跳过太小的轮廓
                if(contour.size() < 5) continue;

                // 绘制原始轮廓（红色）
                cv::drawContours(drawImage, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(0,0,255), 2);

                // 外接矩形（绿色）
                cv::Rect rect = cv::boundingRect(contour);
                cv::rectangle(drawImage, rect, cv::Scalar(0,255,0), 2);

                // 最小外接圆（蓝色）
                cv::Point2f c;
                float r;
                cv::minEnclosingCircle(contour, c, r);
                cv::circle(drawImage, c, r, cv::Scalar(255,0,0), 2);

                // 计算特征参数
                double area = cv::contourArea(contour);
                double perimeter = cv::arcLength(contour, true);
                double aspectRatio = (double)rect.width / rect.height;
                double shapeFactor = (4 * CV_PI * area) / (perimeter * perimeter);
                double sphericity = area / (CV_PI * r * r);

                // 存储参数
                regionParams = QString("宽高比: %1\n形状因子: %2\n球状性: %3")
                              .arg(aspectRatio, 0, 'f', 2)
                              .arg(shapeFactor, 0, 'f', 2)
                              .arg(sphericity, 0, 'f', 2);
            }

            // 显示参数
            if(!regionParams.isEmpty()) {
                ui->lblParameters->setText(regionParams);
                ui->lblParameters->setAlignment(Qt::AlignCenter);
                ui->lblParameters->setWordWrap(true);
                ui->lblParameters->setVisible(true);
            }

            dst = drawImage;
        }}
    };

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
        cv::cvtColor(processedImage, displayProcessed, cv::COLOR_BGR2RGB); // BGR转RGB
    }
    else {
        cv::cvtColor(processedImage, displayProcessed, cv::COLOR_GRAY2RGB); // 灰度转RGB
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

// 事件过滤器
bool MainWindow::eventFilter(QObject *obj, QEvent *event) {
    if (obj == ui->lblImageDisplay_1 && isSelectingROI) {
        QMouseEvent *mouseEvent = dynamic_cast<QMouseEvent*>(event);
        if (!mouseEvent) return false;

        // 计算图像在Label中的显示区域和偏移
        QSize imageSize = originalPixmap.size();
        imageSize.scale(ui->lblImageDisplay_1->size(), Qt::KeepAspectRatio);
        int xOffset = (ui->lblImageDisplay_1->width() - imageSize.width()) / 2;
        int yOffset = (ui->lblImageDisplay_1->height() - imageSize.height()) / 2;
        QPoint adjustedPos = mouseEvent->pos() - QPoint(xOffset, yOffset);

        switch (event->type()) {
            case QEvent::MouseButtonPress: {
                if (adjustedPos.x() < 0 || adjustedPos.y() < 0 ||
                    adjustedPos.x() > imageSize.width() ||
                    adjustedPos.y() > imageSize.height()) {
                    return false;
                }
                this->lassoPoints.clear();
                this->lassoPoints.append(adjustedPos);
                this->isDrawing = true;
                return true;
            }

            case QEvent::MouseMove: {
                if(!this->isDrawing || originalPixmap.isNull()) return false;
                
                QPixmap tempPixmap = originalPixmap.scaled(
                    ui->lblImageDisplay_1->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
                
                QPainter painter(&tempPixmap);
                painter.setPen(QPen(Qt::red, 2, Qt::SolidLine));
                
                // 添加当前点到路径
                this->lassoPoints.append(adjustedPos);
                
                // 绘制套索路径
                if(this->lassoPoints.size() > 1) {
                    painter.drawPolyline(this->lassoPoints.constData(), this->lassoPoints.size());
                }
                
                ui->lblImageDisplay_1->setPixmap(tempPixmap);
                return true;
            }

            case QEvent::MouseButtonRelease: {
                if(this->lassoPoints.size() < 3) {
                    qDebug() << "套索点不足3个，无法创建ROI";
                    this->isDrawing = false;
                    return false;
                }
                
                // 使用原始图像尺寸计算缩放比例
                QSize displayedSize = ui->lblImageDisplay_1->size();
                displayedSize.scale(ui->lblImageDisplay_1->size(), Qt::KeepAspectRatio);
                
                // 修正坐标转换比例计算
                double scaleX = static_cast<double>(originalImage.cols) / imageSize.width();
                double scaleY = static_cast<double>(originalImage.rows) / imageSize.height();
                
                // 转换为图像坐标
                std::vector<cv::Point> contourPoints;
                contourPoints.reserve(this->lassoPoints.size());
                for(const QPoint& pt : std::as_const(this->lassoPoints)) {
                    contourPoints.push_back(cv::Point(
                        qRound(pt.x() * scaleX),
                        qRound(pt.y() * scaleY)
                    ));
                }
                
                // 创建ROI遮罩
                cv::Mat mask = cv::Mat::zeros(originalImage.size(), CV_8UC1);
                std::vector<std::vector<cv::Point>> contours;
                contours.push_back(contourPoints);
                cv::fillPoly(mask, contours, cv::Scalar(255));
                
                // 确保遮罩正确保存
                this->roiMask = mask.clone();
                qDebug() << "遮罩已创建并保存";
                
                
                // 恢复显示状态
                isSelectingROI = false;
                QPixmap scaledPixmap = originalPixmap.scaled(
                    ui->lblImageDisplay_1->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
                ui->lblImageDisplay_1->setPixmap(scaledPixmap);
                
                // 应用ROI
                processImage();
                this->isDrawing = false;
                return true;
            }
            default: break;
        }
    }
    return QMainWindow::eventFilter(obj, event);
}


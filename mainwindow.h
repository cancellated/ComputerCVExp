#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/opencv.hpp>
#include <QDebug>
#include <QFile>
#include <QMessageBox>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_btnHalfInverse_clicked();

    void on_btnEdgeDetect_clicked();

    void on_btnBack_clicked();

    void on_btnSelectFile_clicked();

    void on_btnShapeAnalyze_clicked();

    void on_btnROI_clicked();  // 添加ROI按钮的槽函数声明

private:
    Ui::MainWindow *ui;
    QString currentFunction;
    cv::Mat originalImage;  // 原图
    cv::Mat processedImage; // 处理后的图片
    QString regionParams; 
    void processImage();
    // ROI选择相关成员变量
    QPoint roiStartPoint;
    QPoint roiCurrentPoint;  // 用于跟踪鼠标移动
    QVector<QPoint> lassoPoints; //路径点
    QPixmap originalPixmap;   // 保存原始图像用于重绘
    bool isSelectingROI = false;
    bool isDrawing = false;
    bool eventFilter(QObject *watched, QEvent *event) override;  // 添加事件过滤器声明
    cv::Mat roiMask; // ROI遮罩
};
#endif // MAINWINDOW_H

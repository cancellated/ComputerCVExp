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

private:
    Ui::MainWindow *ui;
    QString currentFunction;
    cv::Mat originalImage;  // 原图
    cv::Mat processedImage; // 处理后的图片
    void processImage();
};
#endif // MAINWINDOW_H

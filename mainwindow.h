#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QKeyEvent>
#include <QDebug>
#include <opencv2/opencv.hpp>
#include "clickablelabel.h"

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
    void on_pushButton_clicked();

private:
    Ui::MainWindow *ui;
private slots:
    void onLabelPressed(QMouseEvent *event);  // Добавь сюда слот
    void onLabelReleased(QMouseEvent *event);
    void onLabelMove(QMouseEvent *event);
    void on_showAugment_clicked();

    void on_blurButton_clicked();

    void on_rotateButton_clicked();

    void on_toHvsButton_clicked();

    void displayImage(cv::Mat img);

    void on_undoButton_clicked();

    QPoint mapToImageCoordinates(const QPoint& widgetPoint);

protected:
    void keyPressEvent(QKeyEvent *event) override;
};
#endif // MAINWINDOW_H

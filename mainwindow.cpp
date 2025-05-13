#include "mainwindow.h"
#include "clickablelabel.h"
#include "./ui_mainwindow.h"
#include <opencv2/opencv.hpp>
#include <QPixmap>
#include <Qimage>
#include <QString>
#include <QDir>
#include <QKeyEvent>
#include <QVBoxLayout>
#include <QPainter>
using namespace cv;
using namespace std;

bool isDrawing = false;
QPoint fPoint;
int maxNumber;
int number;
QString pathToFolder;
QStringList files;
Mat matrixOridin;
Mat lastState;
Mat gray;
Mat augmentPicture;

Mat augmentedImages[7];

QScreen *scrn;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->label->setAlignment(Qt::AlignCenter);
    connect(ui->label, &ClickableLabel::mousePressed, this, &MainWindow::onLabelPressed);
    connect(ui->label, &ClickableLabel::mouseReleased, this, &MainWindow::onLabelReleased);
    connect(ui->label, &ClickableLabel::mouseMove, this, &MainWindow::onLabelMove);

    on_pushButton_clicked();
}

MainWindow::~MainWindow()
{
    delete ui;
}

//переконвертит Mat в Pixmap
static QPixmap matToPixmap(const cv::Mat &mat) {
    if (mat.empty()) {
        return QPixmap();
    }
    if (mat.channels() == 1) {
        return QPixmap::fromImage(QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8));
    } else if (mat.channels() == 3) {
        cv::Mat rgbMat;
        cv::cvtColor(mat, rgbMat, cv::COLOR_BGR2RGB);
        return QPixmap::fromImage(QImage(rgbMat.data, rgbMat.cols, rgbMat.rows, rgbMat.step, QImage::Format_RGB888));
    } else if (mat.channels() == 4) {
        cv::Mat rgbaMat;
        cv::cvtColor(mat, rgbaMat, cv::COLOR_BGRA2RGBA);
        return QPixmap::fromImage(QImage(rgbaMat.data, rgbaMat.cols, rgbaMat.rows, rgbaMat.step, QImage::Format_RGBA8888));
    }

    return QPixmap();
}

void MainWindow:: displayImage(Mat img) {
    QPixmap npxmap = matToPixmap(img);
    QPixmap scaled = npxmap.scaled(scrn->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label->resize(scrn->size());
    ui->label->setPixmap(scaled);
}

Mat blurImage(const cv::Mat& inputImage, int kernelSize = 5) {
    cv::Mat blurred;

    // Убедимся, что ядро нечётное и >= 1
    if (kernelSize % 2 == 0) kernelSize += 1;
    if (kernelSize < 1) kernelSize = 1;

    cv::blur(inputImage, blurred, cv::Size(kernelSize, kernelSize));
    return blurred;
}

Mat bgrToHsv(const cv::Mat& bgrImage) {
    Mat hsvImage;
    CV_Assert(bgrImage.type() == CV_8UC3);
    cv::cvtColor(bgrImage, hsvImage, cv::COLOR_BGR2HSV);
    return hsvImage;
}

QImage createMosaic(int gridRows, int gridCols, int cellWidth, int cellHeight) {
    if (files.isEmpty() || gridRows <= 0 || gridCols <= 0) return QImage();

    QImage mosaic(gridCols * cellWidth, gridRows * cellHeight, QImage::Format_RGB32);
    mosaic.fill(Qt::white);

    QPainter painter(&mosaic);
    painter.setRenderHint(QPainter::SmoothPixmapTransform, true);

    int imgIndex = 0;
    for (int row = 0; row < gridRows; ++row) {
        for (int col = 0; col < gridCols; ++col) {
            if (imgIndex >= files.size())
                break;

            QImage img(pathToFolder + files[imgIndex]);
            if (!img.isNull()) {
                // Произвольный угол поворота (например, 0–360 градусов)
                int angle = rand() % 360;
                QTransform transform;
                transform.rotate(angle);
                QImage rotatedImg = img.transformed(transform, Qt::SmoothTransformation);

                // Обрезка или масштабирование с сохранением произвольных размеров
                QRect targetRect(col * cellWidth, row * cellHeight, cellWidth, cellHeight);
                painter.drawImage(targetRect, rotatedImg);
            }
            ++imgIndex;
        }
    }

    painter.end();
    return mosaic;
}

Mat mosaicImage(const cv::Mat& input, int tileSize = 10) {
    cv::Mat output = input.clone();

    for (int y = 0; y < input.rows; y += tileSize) {
        for (int x = 0; x < input.cols; x += tileSize) {
            int w = std::min(tileSize, input.cols - x);
            int h = std::min(tileSize, input.rows - y);

            cv::Rect tileRect(x, y, w, h);
            cv::Mat tile = input(tileRect);

            // Вычисляем средний цвет тайла
            cv::Scalar avgColor = cv::mean(tile);

            // Заполняем тайл этим цветом
            output(tileRect).setTo(avgColor);
        }
    }

    return output;
}

Mat augmentImage(const Mat &image) {
    augmentedImages[0] = imread((pathToFolder + files[number]).toStdString());
    rotate(image, augmentedImages[0], ROTATE_90_CLOCKWISE);
    rotate(augmentedImages[0], augmentedImages[1], ROTATE_180);
    augmentedImages[2] = imread((pathToFolder + files[number]).toStdString(), 0);
    augmentedImages[3] = bgrToHsv(imread((pathToFolder + files[number]).toStdString()));
    augmentedImages[4] = blurImage(imread((pathToFolder + files[number]).toStdString()));
    flip(imread((pathToFolder + files[number]).toStdString()), augmentedImages[5], 1);
    augmentedImages[6] = mosaicImage(imread((pathToFolder + files[number]).toStdString()));

    return augmentedImages[4];
}

void MainWindow::keyPressEvent(QKeyEvent *event) {
    if (event->key() == Qt::Key_Down) {
        if(number == maxNumber - 1) {
            number = 0;
        }
        else
            number += 1;
        matrixOridin = imread((pathToFolder + files[number]).toStdString());
        displayImage(matrixOridin);
    } else if (event->key() == Qt::Key_Up) {
        if(number == 0) {
            number = maxNumber - 1;
        }
        else
            number -= 1;
        matrixOridin = imread((pathToFolder + files[number]).toStdString());
        displayImage(matrixOridin);
    } else {

    }
    augmentImage(matrixOridin);
}

QPoint labelToImageCoords(const QPoint &labelPoint, const QPixmap &pixmap, QLabel *label) {
    QSize labelSize = label->size();
    QSize pixmapSize = pixmap.size().scaled(labelSize, Qt::KeepAspectRatio);

    int offsetX = (labelSize.width() - pixmapSize.width()) / 2;
    int offsetY = (labelSize.height() - pixmapSize.height()) / 2;

    int x = labelPoint.x() - offsetX;
    int y = labelPoint.y() - offsetY;

    if (x < 0 || y < 0 || x >= pixmapSize.width() || y >= pixmapSize.height())
        return QPoint(-1, -1);

    double scaleX = static_cast<double>(matrixOridin.cols) / pixmapSize.width();
    double scaleY = static_cast<double>(matrixOridin.rows) / pixmapSize.height();

    return QPoint(x * scaleX, y * scaleY);
}

void MainWindow::onLabelPressed(QMouseEvent *event) {
    QPoint imagePoint = labelToImageCoords(event->pos(), ui->label->pixmap(Qt::ReturnByValue), ui->label);
    if (imagePoint.x() < 0 || imagePoint.y() < 0)
        return;

    fPoint = imagePoint;
    isDrawing = true;
    qDebug() << "Начальная точка:" << fPoint;
}

void MainWindow::onLabelMove(QMouseEvent *event) {
    if (!isDrawing) return;

    QPoint lPoint = labelToImageCoords(event->pos(), ui->label->pixmap(Qt::ReturnByValue), ui->label);
    if (lPoint.x() < 0 || lPoint.y() < 0)
        return;

    Mat copyOfOrigin = matrixOridin.clone();
    rectangle(copyOfOrigin, Point(fPoint.x(), fPoint.y()), Point(lPoint.x(), lPoint.y()), Scalar(0, 255, 0), 2);
    displayImage(copyOfOrigin);
}

void MainWindow::onLabelReleased(QMouseEvent *event) {
    if (!isDrawing) return;
    isDrawing = false;

    QPoint lPoint = labelToImageCoords(event->pos(), ui->label->pixmap(Qt::ReturnByValue), ui->label);
    if (lPoint.x() < 0 || lPoint.y() < 0)
        return;

    rectangle(matrixOridin, Point(fPoint.x(), fPoint.y()), Point(lPoint.x(), lPoint.y()), Scalar(139, 0, 255), 2);
    lastState = matrixOridin;
    displayImage(matrixOridin);
}


void MainWindow::on_pushButton_clicked() {
    QDir dir("/Users/andrejradcenko/Desktop/фото");
    pathToFolder = "/Users/andrejradcenko/Desktop/фото/";
    files = dir.entryList(QDir::Files);
    maxNumber = files.length();
    number = 0;
    matrixOridin = imread((pathToFolder + files[number]).toStdString());
    augmentImage(matrixOridin);

    scrn = QGuiApplication::primaryScreen();

    lastState = matrixOridin.clone();
    displayImage(matrixOridin);

    qDebug() << "хуйня";
}

void MainWindow::on_showAugment_clicked()
{
    QImage image = createMosaic(2, 2, 256, 256);
    QPixmap pixMap = QPixmap::fromImage(image);
    QPixmap scaled = pixMap.scaled(ui->label->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label->setPixmap(scaled);

    qDebug() << "EBLAN)))";
}

void MainWindow::on_blurButton_clicked()
{
    lastState = blurImage(lastState);
    displayImage(lastState);
    qDebug() << "залупа";
}

void MainWindow::on_rotateButton_clicked()
{
    rotate(lastState, lastState, ROTATE_90_CLOCKWISE);
    displayImage(lastState);
    qDebug() << "хуй";
}

void MainWindow::on_toHvsButton_clicked()
{
    lastState = bgrToHsv(lastState);
    displayImage(lastState);
    qDebug() << "ZOV";
}



void MainWindow::on_undoButton_clicked()
{
    matrixOridin = imread((pathToFolder + files[number]).toStdString());
    lastState = matrixOridin.clone();
    displayImage(lastState);
}


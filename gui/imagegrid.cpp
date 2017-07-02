#include "imagegrid.h"
#include "ui_imagegrid.h"

#include <QImageReader>
#include <QtWidgets>

ImageGrid::ImageGrid(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ImageGrid)
{
    ui->setupUi(this);
    image = nullptr;
}

bool ImageGrid::loadFile(const QString &fileName)
{
    QImageReader reader(fileName);
    const QImage newImage = reader.read();
    if (newImage.isNull()) {
        QMessageBox::information(this, QGuiApplication::applicationDisplayName(),
                                 tr("Cannot load %1: %2")
                                 .arg(QDir::toNativeSeparators(fileName), reader.errorString()));
        return false;
    }

    setImage(newImage);

    return true;
}

void ImageGrid::setImage(const QImage &newImage)
{
    image = new QImage(newImage);
    ui->imageLabel->setFixedSize(image->size());
    ui->imageLabel->setPixmap(QPixmap::fromImage(*image));
}

void ImageGrid::wheelEvent(QWheelEvent *event)
{
    if (!image) {
        return;
    }

    if (event->delta() > 0) {
        emit scrolledUp(event);
    } else if (event->delta() < 0) {
        emit scrolledDown(event);
    }
}

ImageGrid::~ImageGrid()
{
    delete ui;
    delete image;
}

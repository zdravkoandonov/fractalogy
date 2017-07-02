#ifndef IMAGEGRID_H
#define IMAGEGRID_H

#include <QWidget>

namespace Ui {
class ImageGrid;
}

class ImageGrid : public QWidget
{
    Q_OBJECT

public:
    explicit ImageGrid(QWidget *parent = 0);

    bool loadFile(const QString &);

    ~ImageGrid();

signals:
    void scrolledUp(QWheelEvent *event);
    void scrolledDown(QWheelEvent *event);

private:
    Ui::ImageGrid *ui;
    QImage *image;

    void setImage(const QImage &newImage);
    void wheelEvent(QWheelEvent *event);
};

#endif // IMAGEGRID_H

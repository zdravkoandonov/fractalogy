#ifndef FRACTALOGY_H
#define FRACTALOGY_H

#include <QDir>
#include <QMainWindow>

#include "imagegrid.h"

namespace Ui {
class Fractalogy;
}

class Fractalogy : public QMainWindow
{
    Q_OBJECT

public:
    explicit Fractalogy(QWidget *parent = 0);
    ~Fractalogy();

private slots:
    void on_generateButton_clicked();
    void zoomIn(QWheelEvent *event);
    void zoomOut(QWheelEvent *event);

private:
    Ui::Fractalogy *ui;
    ImageGrid *imageGrid;
    QDir programDir;
    QString program;
    QString imagePath;
    int width, height;
    double lowerX, upperX, lowerY, upperY;
    int threads;

    QStringList getArguments();
    void generateFractalImage();
    void setReadRanges();
};

#endif // FRACTALOGY_H

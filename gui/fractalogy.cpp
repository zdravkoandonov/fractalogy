#include "fractalogy.h"
#include "ui_fractalogy.h"

#include <QDir>
#include <QProcess>
#include <QWheelEvent>

Fractalogy::Fractalogy(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Fractalogy)
{
    ui->setupUi(this);

    imageGrid = new ImageGrid(this);

    connect(imageGrid, SIGNAL(scrolledUp(QWheelEvent*)), this, SLOT(zoomIn(QWheelEvent*)));
    connect(imageGrid, SIGNAL(scrolledDown(QWheelEvent*)), this, SLOT(zoomOut(QWheelEvent*)));

    ui->imageGridContainer->addWidget(imageGrid);

    programDir = QDir::home();
    programDir.cd("Documents/cuda/fractalogy");
    program = programDir.filePath("bin/fractalogy");
}

Fractalogy::~Fractalogy()
{
    delete ui;
    delete imageGrid;
}

void Fractalogy::on_generateButton_clicked()
{
    imagePath = programDir.filePath(ui->filenameInput->text());

    width = ui->widthInput->text().toInt();
    height = ui->heightInput->text().toInt();

    lowerX = ui->lowerXInput->text().toDouble();
    upperX = ui->upperXInput->text().toDouble();
    lowerY = ui->lowerYInput->text().toDouble();
    upperY = ui->upperYInput->text().toDouble();

    threads = ui->threadsInput->text().toInt();

    generateFractalImage();
}

void Fractalogy::zoomIn(QWheelEvent *event)
{
    event->ignore();
    double zoomCoeff = 1.3;

    double x = upperX - lowerX;
    double y = upperY - lowerY;
    double xPos = lowerX + (event->pos().x() / (double)width) * x;
    double yPos = lowerY + (event->pos().y() / (double)height) * y;

    double xPosToLower = xPos - lowerX;
    double xPosToUpper = upperX - xPos;
    double yPosToLower = yPos - lowerY;
    double yPosToUpper = upperY - yPos;

    lowerX = xPos - xPosToLower / zoomCoeff;
    upperX = xPos + xPosToUpper / zoomCoeff;
    lowerX = yPos - yPosToLower / zoomCoeff;
    upperX = yPos + yPosToUpper / zoomCoeff;

    generateFractalImage();
    event->accept();
}

void Fractalogy::zoomOut(QWheelEvent *event)
{
    event->ignore();

    double zoomCoeff = 1.3;

    double x = upperX - lowerX;
    double y = upperY - lowerY;
    double xPos = lowerX + (event->pos().x() / (double)width) * x;
    double yPos = lowerY + (event->pos().y() / (double)height) * y;

    double xPosToLower = xPos - lowerX;
    double xPosToUpper = upperX - xPos;
    double yPosToLower = yPos - lowerY;
    double yPosToUpper = upperY - yPos;

    lowerX = xPos - xPosToLower * zoomCoeff;
    upperX = xPos + xPosToUpper * zoomCoeff;
    lowerX = yPos - yPosToLower * zoomCoeff;
    upperX = yPos + yPosToUpper * zoomCoeff;

    generateFractalImage();
    event->accept();
}

QStringList Fractalogy::getArguments()
{
    QStringList arguments;
    arguments << QString("-s %1x%2").arg(QString::number(width), QString::number(height))
              << QString("-r %1:%2:%3:%4").arg(QString::number(lowerX), QString::number(upperX), QString::number(lowerY), QString::number(upperY))
              << QString("-t %1").arg(QString::number(threads))
              << QString("-o %1").arg(programDir.filePath(imagePath));
    return arguments;
}

void Fractalogy::generateFractalImage()
{
    QProcess process(this);
    process.start(program, getArguments());
    if (!process.waitForFinished()) {
        return;
    }

    imageGrid->loadFile(imagePath);
}

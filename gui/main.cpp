#include "fractalogy.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    Fractalogy w;
    w.show();

    return a.exec();
}

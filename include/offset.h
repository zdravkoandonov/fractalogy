#ifndef OFFSET_H
#define OFFSET_H

struct Offset {
  double lowerX, upperX;
  double lowerY, upperY;

  Offset(double lowerX, double upperX, double lowerY, double upperY)
    : lowerX(lowerX), upperX(upperX), lowerY(lowerY), upperY(upperY) {}
};

#endif

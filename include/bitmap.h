#ifndef BITMAP_H
#define BITMAP_H

#include <stdint.h>

struct Pixel {
  uint8_t red;
  uint8_t green;
  uint8_t blue;
};

struct Bitmap {
  Pixel* pixels;
  size_t width;
  size_t height;

  Bitmap(size_t width, size_t height): pixels(NULL), width(width), height(height) {}
};

#endif

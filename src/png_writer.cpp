#include <png.h>
#include "png_writer.h"

void writePNG(const Bitmap& bitmap, const char* path) {
  png_structp pngPtr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  png_infop infoPtr = png_create_info_struct(pngPtr);
  FILE* fp = fopen(path, "wb");

  size_t width = bitmap.width;
  size_t height = bitmap.height;

  png_set_IHDR(
    pngPtr,
    infoPtr,
    width,
    height,
    8,
    PNG_COLOR_TYPE_RGB,
    PNG_INTERLACE_NONE,
    PNG_COMPRESSION_TYPE_DEFAULT,
    PNG_FILTER_TYPE_DEFAULT
  );

  const int PIXEL_SIZE = 3;
  png_byte** rowPointers = (png_byte **)png_malloc(pngPtr, height * sizeof(png_byte*));

  for (size_t y = 0; y < height; ++y) {
    png_byte* row = (png_byte*)png_malloc(pngPtr, width * PIXEL_SIZE * sizeof(uint8_t));
    rowPointers[y] = row;

    for (size_t x = 0; x < width; ++x) {
      Pixel* pixel = bitmap.pixels + width * y + x;
      *row++ = pixel->red;
      *row++ = pixel->green;
      *row++ = pixel->blue;
    }
  }

  png_init_io(pngPtr, fp);
  png_set_rows(pngPtr, infoPtr, rowPointers);
  png_write_png(pngPtr, infoPtr, PNG_TRANSFORM_IDENTITY, NULL);

  for (size_t y = 0; y < height; ++y) {
    png_free(pngPtr, rowPointers[y]);
  }
  png_free(pngPtr, rowPointers);
}

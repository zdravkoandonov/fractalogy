#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <complex>
#include <iostream>
#include <cmath>
#include <thrust/complex.h>

using thrust::complex;
using thrust::exp;
using thrust::norm;

// A coloured pixel.
struct Pixel {
  uint8_t red;
  uint8_t green;
  uint8_t blue;
};

// A picture.
struct Bitmap {
  Pixel* pixels;
  size_t width;
  size_t height;
};

// Given "bitmap", this returns the pixel of bitmap at the point ("x", "y").
__host__ __device__
Pixel* pixel_at(Bitmap* bitmap, int x, int y) {
  return bitmap->pixels + bitmap->width * y + x;
}

/* Write "bitmap" to a PNG file specified by "path"; returns 0 on
   success, non-zero on error. */
int save_png_to_file(Bitmap* bitmap, const char* path) {
  FILE* fp;
  png_structp png_ptr = NULL;
  png_infop info_ptr = NULL;
  size_t x, y;
  png_byte** row_pointers = NULL;

  /* "status" contains the return value of this function. At first
     it is set to a value which means 'failure'. When the routine
     has finished its work, it is set to a value which means
     'success'. */
  int status = -1;

  /* The following number is set by trial and error only. I cannot
     see where it is documented in the libpng manual. */
  int pixel_size = 3;
  int depth = 8;

  fp = fopen(path, "wb");
  if (!fp) {
    return status;
  }

  png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (png_ptr == NULL) {
    fclose(fp);
  }

  info_ptr = png_create_info_struct(png_ptr);
  if (info_ptr == NULL) {
    png_destroy_write_struct(&png_ptr, &info_ptr);
  }

  /* Set up error handling. */
  if (setjmp(png_jmpbuf(png_ptr))) {
    png_destroy_write_struct(&png_ptr, &info_ptr);
  }

  /* Set image attributes. */
  png_set_IHDR(
    png_ptr,
    info_ptr,
    bitmap->width,
    bitmap->height,
    depth,
    PNG_COLOR_TYPE_RGB,
    PNG_INTERLACE_NONE,
    PNG_COMPRESSION_TYPE_DEFAULT,
    PNG_FILTER_TYPE_DEFAULT
  );

  /* Initialize rows of PNG. */
  row_pointers = (png_byte **)png_malloc(png_ptr, bitmap->height * sizeof(png_byte*));
  for (y = 0; y < bitmap->height; ++y) {
    png_byte* row =
      (png_byte *)png_malloc(png_ptr, bitmap->width * pixel_size * sizeof(uint8_t));
    row_pointers[y] = row;
    for (x = 0; x < bitmap->width; ++x) {
      Pixel* pixel = pixel_at(bitmap, x, y);
      *row++ = pixel->red;
      *row++ = pixel->green;
      *row++ = pixel->blue;
    }
  }

  /* Write the image data to "fp". */
  png_init_io(png_ptr, fp);
  png_set_rows(png_ptr, info_ptr, row_pointers);
  png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

  /* The routine has successfully written the file, so we set
     "status" to a value which indicates success. */
  status = 0;

  for (y = 0; y < bitmap->height; ++y) {
    png_free(png_ptr, row_pointers[y]);
  }
  png_free(png_ptr, row_pointers);

  return status;
}

/* Given "value" and "max", the maximum value which we expect "value"
   to take, this returns an integer between 0 and 255 proportional to
   "value" divided by "max". */
__device__
int rgb_value(int value, int max) {
  return (int)(256 * (value / (double)max));
}

__device__
int iteration(complex<double> c, int limit = 1000) {
  int i = 0;
  double n;
  complex<double> z(0, 0);
  while ((n = norm(z)) < 4 && i < limit) {
    z = exp(z) - c;
    ++i;
  }

  // if (n < 4)
  //   return -1;
  // else
  return i;
}

__global__
void calc(Bitmap* bitmap) {
  // int index = blockIdx.x * blockDim.x + threadIdx.x;
  // int stride = gridDim.x * blockDim.x;

  // for (int i = index; i < n; i += stride) {
  //   result[i] = iteration(c[i]);
  // }

  double lowerX = 0, upperX = 4;
  double lowerY = -2, upperY = 2;

  int x, y;
  int xStride = gridDim.x * blockDim.x;
  int yStride = gridDim.y * blockDim.y;

  complex<double> t;
  int iter;
  Pixel* pixel;
  int width = bitmap->width;
  int height = bitmap->height;
  for (y = blockIdx.y * blockDim.y + threadIdx.y; y < height; y += yStride) {
    for (x = blockIdx.x * blockDim.x + threadIdx.x; x < width; x += xStride) {
      t = complex<double>(lowerX + (upperX - lowerX) * x / (width - 1),
                        lowerY + (upperY - lowerY) * y / (height - 1));
      iter = iteration(t);
      pixel = pixel_at(bitmap, x, y);
      pixel->red = rgb_value(1000 - iter, 1000);
      pixel->green = rgb_value(500 - iter, 1000);
      pixel->blue = rgb_value(200 - iter, 1000);
      // printf("(%d %d) %d (strides: %d %d) (w/h: %d %d) \n", x, y, iter, xStride, yStride, width, height);
    }
    // printf("(%d %d) %d \n", x, y, iter);
  }
}

int main() {
  Bitmap bitmap;
  bitmap.width = 200;
  bitmap.height = 200;
  cudaMalloc(&bitmap.pixels, sizeof(Pixel) * bitmap.width * bitmap.height);

  Bitmap* devBitmap;
  cudaMalloc(&devBitmap, sizeof(Bitmap));
  cudaMemcpy(devBitmap, &bitmap, sizeof(Bitmap), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((bitmap.width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (bitmap.height + threadsPerBlock.y - 1) / threadsPerBlock.y);

  calc<<<numBlocks, threadsPerBlock>>>(devBitmap);

  cudaDeviceSynchronize();

  Pixel* devicePixels = bitmap.pixels;
  bitmap.pixels = new Pixel[bitmap.width * bitmap.height];
  cudaMemcpy(bitmap.pixels, devicePixels, sizeof(Pixel) * bitmap.width * bitmap.height, cudaMemcpyDeviceToHost);

  save_png_to_file(&bitmap, "fractal.png");

  cudaFree(devicePixels);
  cudaFree(devBitmap);

  delete[] bitmap.pixels;

  cudaError error = cudaGetLastError();
  std::cout << cudaGetErrorString(error) << std::endl;

  return 0;
}

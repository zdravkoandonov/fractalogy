#include <stdio.h>
#include <cuComplex.h>
#include <iostream>
#include "offset.h"
#include "bitmap.h"
#include "png_writer.h"

/* Given "value" and "max", the maximum value which we expect "value"
   to take, this returns an integer between 0 and 255 proportional to
   "value" divided by "max". */
__device__
int rgb_value(int value, int max) {
  return (int)(256 * (value / (double)max));
}

__device__
cuDoubleComplex cuCexp(cuDoubleComplex z) {
  double e = exp(cuCreal(z));

  double s, c;
  sincos(cuCimag(z), &s, &c);

  return make_cuDoubleComplex(c * e, s * e);
}

__device__
int iteration(cuDoubleComplex c, int limit = 1000) {
  int i = 0;
  double n;
  cuDoubleComplex z = make_cuDoubleComplex(0, 0);
  while ((n = cuCreal(cuCmul(cuConj(z), z))) < 4 && i < limit) {
    // z = cuCadd(cuCmul(z, z), c);
    // z = cuCsub(cuCexp(z), c);
    // z = c * exp(-z) + z * z;
    z = cuCadd(cuCmul(c, cuCexp(cuCsub(make_cuDoubleComplex(0, 0), z))), cuCmul(z, z));
    ++i;
  }

  // if (n < 4)
  //   return -1;
  // else
  return i;
}

__global__
void calc(Offset offset, Bitmap bitmap) {
  double lowerX = offset.lowerX, upperX = offset.upperX;
  double lowerY = offset.lowerY, upperY = offset.upperY;

  int xStride = gridDim.x * blockDim.x;
  int yStride = gridDim.y * blockDim.y;

  cuDoubleComplex c;
  int iter;
  Pixel* pixel;
  size_t width = bitmap.width;
  size_t height = bitmap.height;

  for (size_t y = blockIdx.y * blockDim.y + threadIdx.y; y < height; y += yStride) {
    for (size_t x = blockIdx.x * blockDim.x + threadIdx.x; x < width; x += xStride) {
      c = make_cuDoubleComplex(lowerX + (upperX - lowerX) * x / (width - 1),
                               lowerY + (upperY - lowerY) * y / (height - 1));
      iter = iteration(c);
      pixel = bitmap.pixels + width * y + x;
      pixel->red = rgb_value(1000 - iter, 1000);
      pixel->green = rgb_value(500 - iter, 1000);
      pixel->blue = rgb_value(200 - iter, 1000);
    }
  }
}

int main(int argc, char** argv) {
  char *svalue = NULL, *rvalue = NULL, *tvalue = NULL, *filenameArg = NULL;

  int c;
  while ((c = getopt(argc, argv, "s:r:t:o:")) != -1)
    switch (c) {
      case 's':
        svalue = optarg;
        break;
      case 'r':
        rvalue = optarg;
        break;
      case 't':
        tvalue = optarg;
        break;
      case 'o':
        filenameArg = optarg;
        break;
      case '?':
        if (optopt == 's')
          fprintf(stderr, "Option -%c requires an argument.\n", optopt);
        else if (isprint(optopt))
          fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        else
          fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
        return 1;
      default:
        abort();
    }

  int width = 640, height = 480;
  double lowerX = -2, upperX = 2;
  double lowerY = -2, upperY = 2;
  int threads = 1;
  char filename[100] = "fractal.png";
  if (svalue != NULL)
    sscanf(svalue, "%dx%d", &width, &height);
  if (rvalue != NULL)
    sscanf(rvalue, "%lf:%lf:%lf:%lf", &lowerX, &upperX, &lowerY, &upperY);
  if (tvalue != NULL)
    sscanf(tvalue, "%d", &threads);
  if (filenameArg != NULL)
    sscanf(filenameArg, "%s", filename);
  printf("svalue = %s;%dx%d\nrvalue = %s; %lf, %lf, %lf, %lf\n", svalue, width, height, rvalue, lowerX, upperX, lowerY, upperY);
  printf("tvalue = %s; %d\n", tvalue, threads);
  printf("filenameArg = %s; %s\n", filenameArg, filename);

  //////////////////////////////////

  Offset offset(lowerX, upperX, lowerY, upperY);
  Bitmap bitmap(width, height);

  size_t pixelsCount = bitmap.width * bitmap.height;
  size_t pixelsSize = pixelsCount * sizeof(Pixel);
  cudaMalloc(&bitmap.pixels, pixelsSize);

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((bitmap.width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (bitmap.height + threadsPerBlock.y - 1) / threadsPerBlock.y);

  calc<<<numBlocks, threadsPerBlock>>>(offset, bitmap);

  cudaDeviceSynchronize();

  Pixel* devicePixels = bitmap.pixels;
  bitmap.pixels = new Pixel[pixelsCount];
  cudaMemcpy(bitmap.pixels, devicePixels, pixelsSize, cudaMemcpyDeviceToHost);

  writePNG(bitmap, filename);

  cudaFree(devicePixels);
  delete[] bitmap.pixels;

  cudaError error = cudaGetLastError();
  std::cout << cudaGetErrorString(error) << std::endl;

  return 0;
}

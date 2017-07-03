#include <stdio.h>
#include <iostream>
#include <cuComplex.h>
#include <getopt.h>
#include <cmath>
#include <sys/time.h>
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
void calc(Offset offset, Bitmap bitmap, bool quiet) {
  if (!quiet)
    printf("Thread-%d:%d on block %d:%d started.\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);

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

  if (!quiet)
    printf("Thread-%d:%d on block %d:%d finished.\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
}

void threadConfigCalc(int maxThreads, int &threadsPerBlock1dim, int &numBlocksX, int &numBlocksY, bool quiet = false) {
  double sqroot = sqrt(maxThreads);
  threadsPerBlock1dim = exp2(fmin(floor(log2(sqroot)), 4));
  numBlocksY = floor(sqroot / threadsPerBlock1dim);
  numBlocksX = floor((double)maxThreads / (threadsPerBlock1dim * threadsPerBlock1dim * numBlocksY));

  // if (!quiet)
    printf("Threads used in current run: %d <= %d, threadsPerBlock: %dx%d, numBlocksXxnumBlocksY: %dx%d\n",
      threadsPerBlock1dim * threadsPerBlock1dim * numBlocksX * numBlocksY, maxThreads, threadsPerBlock1dim, threadsPerBlock1dim, numBlocksX, numBlocksY);
}

double cpuSecondMonolitic() {
  struct timespec tp;
  clock_gettime(CLOCK_MONOTONIC, &tp);
  return ((double)tp.tv_sec + (double)tp.tv_nsec*1.e-9);
}

void generateImage(int width, int height, Offset offset, int maxThreads, const char* filename, bool quiet) {
  Bitmap bitmap(width, height);

  size_t pixelsCount = bitmap.width * bitmap.height;
  size_t pixelsSize = pixelsCount * sizeof(Pixel);
  cudaMalloc(&bitmap.pixels, pixelsSize);

  int threadsPerBlock1dim, numBlocksX, numBlocksY;
  threadConfigCalc(maxThreads, threadsPerBlock1dim, numBlocksX, numBlocksY, quiet);

  dim3 threadsPerBlock(threadsPerBlock1dim, threadsPerBlock1dim);
  dim3 numBlocks(numBlocksX, numBlocksY);

  // TIMINGS
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  calc<<<numBlocks, threadsPerBlock>>>(offset, bitmap, quiet);
  cudaEventRecord(stop);
  cudaDeviceSynchronize();

  Pixel* devicePixels = bitmap.pixels;
  bitmap.pixels = new Pixel[pixelsCount];
  cudaMemcpy(bitmap.pixels, devicePixels, pixelsSize, cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  // if (!quiet)
    printf("Execution time on gpu: %f\n", milliseconds / 1000);

  writePNG(bitmap, filename);

  cudaFree(devicePixels);
  delete[] bitmap.pixels;

  cudaError error = cudaGetLastError();
  if (!quiet)
    printf("%s\n", cudaGetErrorString(error));
}

int main(int argc, char** argv) {
  double iStart = cpuSecondMonolitic();

  int width = 640, height = 480;
  double lowerX = -2, upperX = 2;
  double lowerY = -2, upperY = 2;
  int maxThreads = 1;
  char filename[100] = "fractal.png";
  bool quiet = false;

  char *svalue = NULL, *rvalue = NULL, *tvalue = NULL, *filenameArg = NULL;
  int c;
  static struct option long_options[] =
      {
        {"quiet",   no_argument,       0, 'q'},
        {"size",    required_argument, 0, 's'},
        {"rect",    required_argument, 0, 'r'},
        {"tasks",   required_argument, 0, 't'},
        {"output",  required_argument, 0, 'o'},
        {0, 0, 0, 0}
      };

  while ((c = getopt_long_only(argc, argv, "s:r:t:o:q", long_options, NULL)) != -1)
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
      case 'q':
        quiet = true;
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

  if (svalue != NULL)
    sscanf(svalue, "%dx%d", &width, &height);
  if (rvalue != NULL)
    sscanf(rvalue, "%lf:%lf:%lf:%lf", &lowerX, &upperX, &lowerY, &upperY);
  if (tvalue != NULL)
    sscanf(tvalue, "%d", &maxThreads);
  if (filenameArg != NULL)
    sscanf(filenameArg, "%s", filename);

  if (!quiet) {
    printf("svalue = %s;%dx%d\nrvalue = %s; %lf, %lf, %lf, %lf\n", svalue, width, height, rvalue, lowerX, upperX, lowerY, upperY);
    printf("tvalue = %s; %d\n", tvalue, maxThreads);
    printf("filenameArg = %s; %s\n", filenameArg, filename);
    printf("quiet = %d\n", quiet);
  }
  //////////////////////////////////

  Offset offset(lowerX, upperX, lowerY, upperY);
  generateImage(width, height, offset, maxThreads, filename, quiet);

  double iElaps = cpuSecondMonolitic() - iStart;
  // if (!quiet)
    printf("Total execution time for this run: %lf\n", iElaps);

  return 0;
}

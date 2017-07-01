CC = nvcc
CFLAGS = -gencode arch=compute_20,code=sm_20

fractology:
	$(CC) $(CFLAGS) src/png.cu -lpng -o bin/png

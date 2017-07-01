CC = nvcc
CFLAGS = -gencode arch=compute_20,code=sm_20

fractology:
	$(CC) $(CFLAGS) src/fractalogy.cu -lpng -o bin/png

CC = nvcc
CFLAGS = -gencode arch=compute_13,code=sm_13 -Iinclude

fractalogy:
	$(CC) $(CFLAGS) src/*.cu src/*.cpp -lpng -o bin/fractalogy

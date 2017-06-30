CC = nvcc
CFLAGS = -gencode arch=compute_20,code=sm_20

IDIR = ./include
SDIR = ./src
ODIR = ./build
EDIR = ./bin

_DEPS =
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJS = fractalogy.o
OBJS = $(patsubst %,$(ODIR)/%,$(_OBJS))

fractalogy: $(OBJS)
	$(CC) $(CFLAGS) $^ -o $(EDIR)/$@

$(ODIR)/%.o: $(SDIR)/%.cpp $(DEPS)
	$(CC) -x cu $(CFLAGS) -I$(IDIR) -dc $< -o $@

clean:
	rm -f $(ODIR)/*.o $(EDIR)/*

.PHONY: clean

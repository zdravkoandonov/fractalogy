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

$(EDIR)/fractalogy: $(OBJS) | $(EDIR)
	$(CC) $(CFLAGS) $^ -o $@

$(ODIR)/%.o: $(SDIR)/%.cpp $(DEPS) | $(ODIR)
	$(CC) -x cu $(CFLAGS) -I$(IDIR) -dc $< -o $@

$(ODIR):
	mkdir $@

$(EDIR):
	mkdir $@

clean:
	rm -f $(ODIR)/*.o $(EDIR)/*

.PHONY: clean

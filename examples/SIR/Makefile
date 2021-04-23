include ../../Makefile.in

C = main.cc rng.cc history.cc simfunctions.cc
H = smctc.hh sampler.hh particle.hh moveset.hh history.hh rng.hh

PFC = SIR_example.cc SIR_funcs.cc 
PFO = SIR_example.o SIR_funcs.o
PFH = SIR_funcs.hh

CCFLAGS := $(CCFLAGS) -I../../include -L../../lib 
LFLAGS := -I../../include -L../../lib $(LFLAGS)

all: pf

pf: $(PFC) $(PFH)
	$(CC) $(CCFLAGS) -g -c $(PFC)
	$(CC) $(PFO) -lsmctc $(LFLAGS) -opf
	cp pf ../../bin
	cp data.csv ../../bin

clean:
	-rm *.o
	-rm *~
	-rm pf
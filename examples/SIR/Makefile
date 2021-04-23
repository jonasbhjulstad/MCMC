include ../../Makefile.in

C = main.cc rng.cc history.cc simfunctions.cc
H = smctc.hh sampler.hh particle.hh moveset.hh history.hh rng.hh

PFC = SIR_example.cc SIR_funcs.cc 
PFO = SIR_example.o SIR_funcs.o
PFH = SIR_funcs.hh


CCFLAGS := $(CCFLAGS) -I../../include -L../../lib 
LFLAGS := -I../../include -L../../lib $(LFLAGS)

all: SIR

SIR: $(PFC) $(PFH)
	$(CC) $(CCFLAGS) -c $(PFC)
	$(CC) $(PFO) -lsmctc $(LFLAGS) -o SIR
	cp SIR ../../bin
	cp data.csv ../../bin

debug: $(PFC) $(PFH)
	$(CC) -g $(CCFLAGS) -c $(PFC)
	$(CC) $(PFO) -lsmctc $(LFLAGS) -o SIR

clean:
	-rm *.o
	-rm *~
	-rm SIR
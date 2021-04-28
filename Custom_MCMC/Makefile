
CC = g++

C = main.cc simfunctions.cc
O = main.o simfunctions.o
H = simfunctions.hh

CCFLAGS := $(CCFLAGS) -I include -L lib
LFLAGS := -g -I include -L lib -static $(LFLAGS)


all: MCMC

.PHONY: clean

clean:
	-rm *.o
	-rm *~
	-rm MCMC

MCMC: $(C) $(H)
	$(CC) $(CCFLAGS) -g -c $(C)
	$(CC) $(O) -lsmctc -lgsl -lgslcblas -lm $(LFLAGS) -o MCMC



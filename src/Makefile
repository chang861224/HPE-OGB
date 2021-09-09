CC = g++
CPPFLAGS = -std=c++11 -fPIC -fopenmp -lm -Ofast
EXE = HPE BPR
OBJS1 = HPE.o HPE_cli.o BPR.o BPR_cli.o
OBJS2 = ../smore/src/proNet.o ../smore/src/random.o ../smore/src/util.o ../smore/src/model/LINE.o

all : $(EXE)

$(EXE) : $(OBJS1)
	$(CC) $(CPPFLAGS) -o ../$@ $@.o $@_cli.o $(OBJS2)

%.o : %.cpp
	$(CC) $(CPPFLAGS) -c -o $@ $<

clean :
	( cd ../; rm $(EXE) )
	rm $(OBJS1)

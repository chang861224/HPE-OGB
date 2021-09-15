CC = g++
CPPFLAGS = -std=c++11 -fPIC -fopenmp -lm -Ofast
EXE = HPE BPR HOPREC
OBJS1 = command-line/HPE.o command-line/BPR.o command-line/HOPREC.o
OBJS2 = model/HPE.o model/BPR.o model/HOPREC.o
OBJS3 = ../smore/src/proNet.o ../smore/src/random.o ../smore/src/util.o ../smore/src/model/LINE.o

all : $(EXE)

$(EXE) : $(OBJS1) $(OBJS2)
	$(CC) $(CPPFLAGS) -o ../$@ model/$@.o command-line/$@.o $(OBJS3)
#$(CC) $(CPPFLAGS) -o ../$@ $@.o $@_cli.o $(OBJS3)

%.o : %.cpp
	$(CC) $(CPPFLAGS) -c -o $@ $<

clean :
	( cd ../; rm $(EXE) )
	rm $(OBJS1) $(OBJS2)

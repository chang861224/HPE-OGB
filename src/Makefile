CC = g++
CPPFLAGS = -std=c++11 -fPIC -fopenmp -lm -Ofast
EXE = ../hpe
OBJS1 = HPE.o HPE_cli.o
OBJS2 = ../smore/src/proNet.o ../smore/src/random.o ../smore/src/util.o ../smore/src/model/LINE.o

all : $(OBJS1)
	$(CC) $(CPPFLAGS) -o $(EXE) $(OBJS1) $(OBJS2)

%.o : %.cpp
	$(CC) $(CPPFLAGS) -c -o $@ $<

clean :
	rm $(EXE) $(OBJS1)

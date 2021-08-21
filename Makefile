CC=g++
CPPFLAGS = -std=c++11 -fPIC -fopenmp -lm -Ofast

hpe : ./src/HPE_cli.o ./src/HPE.o ./smore/src/proNet.o ./smore/src/random.o ./smore/src/util.o ./smore/src/model/LINE.o
	$(CC) $(CPPFLAGS) -o hpe ./src/HPE_cli.o ./src/HPE.o ./smore/src/proNet.o ./smore/src/random.o ./smore/src/util.o ./smore/src/model/LINE.o

HPE_cli.o : ./src/HPE_cli.cpp ./src/HPE.h
	cd ./src/
	$(CC) $(CPPFLAGS) -c ./src/HPE_cli.cpp ./src/HPE.h
	cd ../

HPE.o : ./src/HPE.cpp ./src/HPE.h
	cd ./src/
	$(CC) $(CPPFLAGS) -c ./src/HPE.cpp ./src/HPE.h
	cd ../

clean :
	rm ./src/*.o

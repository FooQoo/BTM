CC=g++
CFLAGS=-c -pipe -O3
MODEL_OBJ=model.o infer.o main.o
MODEL_EXE=btm
CPPFLAGS=-I/usr/local/Cellar/boost/1.67.0_1/include
LDFLAGS=-L/usr/local/Cellar/boost/1.67.0_1/lib/
LDLIBS=-lboost_math_tr1f-mt

all:$(MODEL_EXE) 
$(MODEL_EXE):$(MODEL_OBJ)
	$(CC) $(MODEL_OBJ) -o $@

main.o:model.h
model.o:model.h biterm.h doc.h
infer.o:infer.h doc.h

%.o:%.cpp
	$(CC) $(CPPFLAGS) $(LDFLAGS) $(INCLUDE) $(CFLAGS) $(LDLIBS)  $< -o $@

clean:
	rm -rf $(MODEL_OBJ) $(MODEL_EXE)

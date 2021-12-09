all:
	make swigCall

swigCall:
	swig -python -c++ swigCall.i
	g++ -c swigCall.cpp swigCall_wrap.cxx -I/home/zongzan/anaconda3/envs/STR36/include/python3.6m/ -fPIC
	g++ -shared swigCall.o swigCall_wrap.o -o _swigCall.so

clean:
	rm -f swigCall_wrap.c*
	rm -f *.py*
	rm -f *.so
	rm -f *.o
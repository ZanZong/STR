# build instead of setup tools
# gcc -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -DMAJOR_VERSION=1 -DMINOR_VERSION=0 -I/home/zongzan/anaconda3/envs/STR36/include -I/home/zongzan/anaconda3/envs/STR36/include/python3.6m/ -c great_module.cc -o great_module.o
# gcc -shared great_module.o -L/home/zongzan/anaconda3/envs/STR36/lib -o great_module.so

# for make file
mkdir build
cd build
cmake ..
make -j3
mv ./libtest_module.so ../_test_module.so
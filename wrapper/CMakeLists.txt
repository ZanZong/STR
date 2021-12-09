cmake_minimum_required(VERSION 2.8.3)
project(test)
add_compile_options(-std=c++11)
include_directories(
    /home/zongzan/anaconda3/envs/STR36/include/python3.6m/
)
add_library(test_module SHARED swigtest.cc test_module_wrap.cxx)

#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Method1, <Python.h> is not requred
// extern "C" {
// long function(long a) {
//     return a + 1;
// }
// }

// Method2, using extensions in python.h, 
long long fibonacci(unsigned int n) {     //C语言实现的斐波那契函数
    if (n < 2) {
        return 1;
    } else {
        return fibonacci(n-2) + fibonacci(n-1);
    }
    //Py_RETURN_NONE;  无返回值的情况
}

static PyObject* fibonacci_py(PyObject* self, PyObject* args) {//Cpython解释器可以识别的C函数
    PyObject *result = NULL;   //包含异常处理
    long n;
    if (PyArg_ParseTuple(args, "l", &n)) {//l代表希望args是长整型
        result = Py_BuildValue("L", fibonacci((unsigned int)n)); //unisgned int避免附属
    }
    return result;
}


static char fibonacci_docs[] =
    "fibonacci(n): Return nth Fibonacci sequence number "
    "computed recursively\n";


static PyMethodDef fibonacci_module_methods[] = { //方法映射表
    {"fibonacci" /*暴露给python的函数名*/, (PyCFunction)fibonacci_py/*函数指针，真正函数的定义的地方*/,
     METH_VARARGS/*告诉python解析器想用三种函数签名二点哪一种（METH_VARARGS,MET_KEYWORDSMET_KEYWORDS关键字参数,MET_NOARGS无参数*/, fibonacci_docs/*函数文档字符串*/},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef fibonacci_module_definition = { 
    PyModuleDef_HEAD_INIT,
    "fibonacci",   //被导出的模块名
    "Extension module that provides fibonacci sequence function",  //模块注释
    -1,
    fibonacci_module_methods    //方法映射表
};


PyMODINIT_FUNC PyInit_fibonacci(void) {  //初始化函数，在模块被导入时被python解析器调用 唯一一个非static函数
    Py_Initialize();
    return PyModule_Create(&fibonacci_module_definition);
}
// 最后使用setuptools打包配置就可以在python程序中调用该模块。
/*
from setuptools import setup, Extension
setup(
    name='fibonacci',
    ext_modules=[
    Extension('fibonacci',['fibonacci.c']),
    ]
)

编译
$ python setup.py install
setup会调用
gcc -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -DMAJOR_VERSION=1 -DMINOR_VERSION=0 -I/home/zongzan/anaconda3/envs/STR36/include -I/home/zongzan/anaconda3/envs/STR36/include/python3.6m/ -c great_module.cc -o great_module.o
gcc -shared great_module.o -L/home/zongzan/anaconda3/envs/STR36/lib -o great_module.so
*/
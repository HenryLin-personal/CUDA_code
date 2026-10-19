#include <pybind11/pybind11.h>
// #include <ATen/ATen.h>
// #include <torch/library.h>
#include <iostream>


// #include <Python.h>
// 1. C++函数
void myhello()
{
    printf("hello setuptools from cpp backend!\n");
}

// // 2. 给 Python C API 用的包装函数（把 void 转成 Python 的 None 返回）
// static PyObject* py_myhello(PyObject* self, PyObject* args) {
//     myhello();               // 调用你的 C++ 函数
//     Py_RETURN_NONE;          // 相当于 Python 里的 return None
// }

// // 3. 定义模块的函数表（把名字 "myhello" 和函数指针绑定起来）
// static PyMethodDef MyMethods[] = {
//     {"myhello", py_myhello, METH_NOARGS, "Print hello from C++"},
//     {NULL, NULL, 0, NULL}    // 结束标志
// };

// // 4. 在 PyInit__C 中把最后一个 NULL 换成上面的 MyMethods
// extern "C" PyObject* PyInit__C(void) {
//   static struct PyModuleDef def = {
//     PyModuleDef_HEAD_INIT, "_C", NULL, -1, 
//     Mymethods
//     };
//   return PyModule_Create(&def);
// }


/* 直接使用pybind11省去上面的接口 */
PYBIND11_MODULE(_C, m) {
    m.def("myhello", &myhello, "Print hello from C++");
}












// at::Tensor myadd_cpu(const at::Tensor& a, const at::Tensor& b, double alpha) {
//   TORCH_CHECK(a.sizes() == b.sizes());
//   return a + alpha * b;  // 先用 ATen 自带的运算，跑通注册流程以后再换成手写循环
// }

// TORCH_LIBRARY(myhello, m) {
//   m.def("myadd(Tensor a, Tensor b, float alpha) -> Tensor");
// }
// TORCH_LIBRARY_IMPL(myhello, CPU, m) {
//   m.impl("myadd", &myadd_cpu);
// }
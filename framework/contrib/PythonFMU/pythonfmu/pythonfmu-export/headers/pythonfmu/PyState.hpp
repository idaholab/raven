
#ifndef PYTHONFMU_PYTHONSTATE_HPP
#define PYTHONFMU_PYTHONSTATE_HPP

#include <Python.h>
#include <iostream>

namespace pythonfmu
{

class PyState
{
public:
    PyState()
    {
        was_initialized_ = Py_IsInitialized();

        if (!was_initialized_) {
            Py_SetProgramName(L"./PythonFMU");
            Py_Initialize();
            PyEval_InitThreads();
            _mainPyThread = PyEval_SaveThread();
        }
    }

    ~PyState()
    {
        if (!was_initialized_) {
            PyEval_RestoreThread(_mainPyThread);
            Py_Finalize();
        }
    }

private:
    bool was_initialized_;
    PyThreadState* _mainPyThread;
};

} // namespace pythonfmu

#endif //PYTHONFMU_PYTHONSTATE_HPP


#include "pythonfmu/PySlaveInstance.hpp"

#include "pythonfmu/PyState.hpp"

#include "cppfmu/cppfmu_cs.hpp"

#include <fstream>
#include <functional>
#include <regex>
#include <sstream>
#include <utility>

namespace pythonfmu
{

inline std::string getline(const std::string& fileName)
{
    std::string line;
    std::ifstream infile(fileName);
    std::getline(infile, line);
    return line;
}

inline std::string findClassName(const std::string& fileName)
{
    std::string line;
    std::ifstream infile(fileName);
    std::string regexStr(R"(^class (\w+)\(\s*Fmi2Slave\s*\)\s*:)");
    while (getline(infile, line)) {
        std::smatch m;
        std::regex re(regexStr);
        if (std::regex_search(line, m, re)) {
            return m[1];
        }
    }
    return "";
}

inline void py_safe_run(const std::function<void(PyGILState_STATE gilState)>& f)
{
    PyGILState_STATE gil_state = PyGILState_Ensure();
    f(gil_state);
    PyGILState_Release(gil_state);
}

PySlaveInstance::PySlaveInstance(std::string instanceName, std::string resources, const cppfmu::Logger& logger, const bool visible)
    : instanceName_(std::move(instanceName))
    , resources_(std::move(resources))
    , logger_(logger)
    , visible_(visible)
{
    py_safe_run([this](PyGILState_STATE gilState) {
        // Append resources path to python sys path
        PyObject* sys_module = PyImport_ImportModule("sys");
        if (sys_module == nullptr) {
            handle_py_exception("[ctor] PyImport_ImportModule", gilState);
        }
        PyObject* sys_path = PyObject_GetAttrString(sys_module, "path");
        Py_DECREF(sys_module);
        if (sys_path == nullptr) {
            handle_py_exception("[ctor] PyObject_GetAttrString", gilState);
        }
        int success = PyList_Insert(sys_path, 0, PyUnicode_FromString(resources_.c_str()));
        Py_DECREF(sys_path);
        if (success != 0) {
            handle_py_exception("[ctor] PyList_Insert", gilState);
        }

        std::string moduleName = getline(resources_ + "/slavemodule.txt");
        PyObject* pModule = PyImport_ImportModule(moduleName.c_str());
        if (pModule == nullptr) {
            handle_py_exception("[ctor] PyImport_ImportModule", gilState);
        }

        std::string className = findClassName(resources_ + "/" + moduleName + ".py");
        if (className.empty()) {
            cleanPyObject();
            throw cppfmu::FatalError("Unable to find class extending Fmi2Slave!");
        }

        PyObject* pyClassName = Py_BuildValue("s", className.c_str());
        if (pyClassName == nullptr) {
            handle_py_exception("[ctor] Py_BuildValue", gilState);
        }

        pClass_ = PyObject_GetAttr(pModule, pyClassName);
        Py_DECREF(pyClassName);
        Py_DECREF(pModule);
        if (pClass_ == nullptr) {
            handle_py_exception("[ctor] PyObject_GetAttr", gilState);
        }

        initialize(gilState);
    });
}

void PySlaveInstance::clearLogBuffer() const
{
    clearLogStrBuffer();

    if (pMessages_ == NULL) {
      return;
    }

    PyObject* debugField = Py_BuildValue("s", "debug");
    PyObject* msgField = Py_BuildValue("s", "msg");
    PyObject* categoryField = Py_BuildValue("s", "category");
    PyObject* statusField = Py_BuildValue("s", "status");

    auto size = PyList_Size(pMessages_);
    for (auto i = 0; i < size; i++) {
        PyObject* msg = PyList_GetItem(pMessages_, i);

        auto debugAttr = PyObject_GetAttr(msg, debugField);
        auto msgAttr = PyObject_GetAttr(msg, msgField);
        auto categoryAttr = PyObject_GetAttr(msg, categoryField);
        auto statusAttr = PyObject_GetAttr(msg, statusField);

        auto statusValue = static_cast<cppfmu::FMIStatus>(PyLong_AsLong(statusAttr));

        PyObject* msgValue = PyUnicode_AsEncodedString(msgAttr, "utf-8", nullptr);
        char* msgStr = PyBytes_AsString(msgValue);
        logStrBuffer.emplace_back(msgValue);

        const char* categoryStr = "";
        if (categoryAttr != Py_None) {
            PyObject* categoryValue = PyUnicode_AsEncodedString(categoryAttr, "utf-8", nullptr);
            categoryStr = PyBytes_AsString(categoryValue);
            logStrBuffer.emplace_back(categoryValue);
        }

        if (PyObject_IsTrue(debugAttr)) {
            const_cast<cppfmu::Logger&>(logger_).DebugLog(statusValue, categoryStr, msgStr);
        } else {
            const_cast<cppfmu::Logger&>(logger_).Log(statusValue, categoryStr, msgStr);
        }
    }
    Py_DECREF(debugField);
    Py_DECREF(msgField);
    Py_DECREF(categoryField);
    Py_DECREF(statusField);
    PyList_SetSlice(pMessages_, 0, size, nullptr);
}

void PySlaveInstance::initialize(PyGILState_STATE gilState)
{
    Py_XDECREF(pInstance_);
    Py_XDECREF(pMessages_);

    PyObject* args = PyTuple_New(0);
    PyObject* kwargs = Py_BuildValue("{ss,ss,sL,si}",
        "instance_name", instanceName_.c_str(),
        "resources", resources_.c_str(),
        "logger", &logger_,
        "visible", visible_);
    pInstance_ = PyObject_Call(pClass_, args, kwargs);
    Py_DECREF(args);
    Py_DECREF(kwargs);
    if (pInstance_ == nullptr) {
        handle_py_exception("[initialize] PyObject_Call", gilState);
    }
    pMessages_ = PyObject_CallMethod(pInstance_, "_get_log_queue", nullptr);
}

void PySlaveInstance::SetupExperiment(cppfmu::FMIBoolean, cppfmu::FMIReal, cppfmu::FMIReal startTime, cppfmu::FMIBoolean, cppfmu::FMIReal)
{
    py_safe_run([this, startTime](PyGILState_STATE gilState) {
        auto f = PyObject_CallMethod(pInstance_, "setup_experiment", "(d)", startTime);
        if (f == nullptr) {
            handle_py_exception("[setupExperiment] PyObject_CallMethod", gilState);
        }
        Py_DECREF(f);
        clearLogBuffer();
    });
}

void PySlaveInstance::EnterInitializationMode()
{
    py_safe_run([this](PyGILState_STATE gilState) {
        auto f = PyObject_CallMethod(pInstance_, "enter_initialization_mode", nullptr);
        if (f == nullptr) {
            handle_py_exception("[enterInitializationMode] PyObject_CallMethod", gilState);
        }
        Py_DECREF(f);
        clearLogBuffer();
    });
}

void PySlaveInstance::ExitInitializationMode()
{
    py_safe_run([this](PyGILState_STATE gilState) {
        auto f = PyObject_CallMethod(pInstance_, "exit_initialization_mode", nullptr);
        if (f == nullptr) {
            handle_py_exception("[exitInitializationMode] PyObject_CallMethod", gilState);
        }
        Py_DECREF(f);
        clearLogBuffer();
    });
}

bool PySlaveInstance::DoStep(cppfmu::FMIReal currentTime, cppfmu::FMIReal stepSize, cppfmu::FMIBoolean, cppfmu::FMIReal& endOfStep)
{
    bool status;
    py_safe_run([this, &status, currentTime, stepSize](PyGILState_STATE gilState) {
        auto f = PyObject_CallMethod(pInstance_, "do_step", "(dd)", currentTime, stepSize);
        if (f == nullptr) {
            handle_py_exception("[doStep] PyObject_CallMethod", gilState);
        }
        status = static_cast<bool>(PyObject_IsTrue(f));
        Py_DECREF(f);
        clearLogBuffer();
    });

    return status;
}

void PySlaveInstance::Reset()
{
    py_safe_run([this](PyGILState_STATE gilState) {
        initialize(gilState);
    });
}

void PySlaveInstance::Terminate()
{
    py_safe_run([this](PyGILState_STATE gilState) {
        auto f = PyObject_CallMethod(pInstance_, "terminate", nullptr);
        if (f == nullptr) {
            handle_py_exception("[terminate] PyObject_CallMethod", gilState);
        }
        Py_DECREF(f);
        clearLogBuffer();
    });
}

void PySlaveInstance::SetReal(const cppfmu::FMIValueReference* vr, std::size_t nvr, const cppfmu::FMIReal* values)
{
    py_safe_run([this, &vr, nvr, &values](PyGILState_STATE gilState) {
        PyObject* vrs = PyList_New(nvr);
        PyObject* refs = PyList_New(nvr);
        for (int i = 0; i < nvr; i++) {
            PyList_SetItem(vrs, i, Py_BuildValue("i", vr[i]));
            PyList_SetItem(refs, i, Py_BuildValue("d", values[i]));
        }

        auto f = PyObject_CallMethod(pInstance_, "set_real", "(OO)", vrs, refs);
        Py_DECREF(vrs);
        Py_DECREF(refs);
        if (f == nullptr) {
            handle_py_exception("[setReal] PyObject_CallMethod", gilState);
        }
        Py_DECREF(f);
        clearLogBuffer();
    });
}

void PySlaveInstance::SetInteger(const cppfmu::FMIValueReference* vr, std::size_t nvr, const cppfmu::FMIInteger* values)
{
    py_safe_run([this, &vr, nvr, &values](PyGILState_STATE gilState) {
        PyObject* vrs = PyList_New(nvr);
        PyObject* refs = PyList_New(nvr);
        for (int i = 0; i < nvr; i++) {
            PyList_SetItem(vrs, i, Py_BuildValue("i", vr[i]));
            PyList_SetItem(refs, i, Py_BuildValue("i", values[i]));
        }

        auto f = PyObject_CallMethod(pInstance_, "set_integer", "(OO)", vrs, refs);
        Py_DECREF(vrs);
        Py_DECREF(refs);
        if (f == nullptr) {
            handle_py_exception("[setInteger] PyObject_CallMethod", gilState);
        }
        Py_DECREF(f);
        clearLogBuffer();
    });
}

void PySlaveInstance::SetBoolean(const cppfmu::FMIValueReference* vr, std::size_t nvr, const cppfmu::FMIBoolean* values)
{
    py_safe_run([this, &vr, nvr, &values](PyGILState_STATE gilState) {
        PyObject* vrs = PyList_New(nvr);
        PyObject* refs = PyList_New(nvr);
        for (int i = 0; i < nvr; i++) {
            PyList_SetItem(vrs, i, Py_BuildValue("i", vr[i]));
            PyList_SetItem(refs, i, PyBool_FromLong(values[i]));
        }

        auto f = PyObject_CallMethod(pInstance_, "set_boolean", "(OO)", vrs, refs);
        Py_DECREF(vrs);
        Py_DECREF(refs);
        if (f == nullptr) {
            handle_py_exception("[setBoolean] PyObject_CallMethod", gilState);
        }
        Py_DECREF(f);
        clearLogBuffer();
    });
}

void PySlaveInstance::SetString(const cppfmu::FMIValueReference* vr, std::size_t nvr, cppfmu::FMIString const* values)
{
    py_safe_run([this, &vr, nvr, &values](PyGILState_STATE gilState) {
        PyObject* vrs = PyList_New(nvr);
        PyObject* refs = PyList_New(nvr);
        for (int i = 0; i < nvr; i++) {
            PyList_SetItem(vrs, i, Py_BuildValue("i", vr[i]));
            PyList_SetItem(refs, i, Py_BuildValue("s", values[i]));
        }

        auto f = PyObject_CallMethod(pInstance_, "set_string", "(OO)", vrs, refs);
        Py_DECREF(vrs);
        Py_DECREF(refs);
        if (f == nullptr) {
            handle_py_exception("[setString] PyObject_CallMethod", gilState);
        }
        Py_DECREF(f);
        clearLogBuffer();
    });
}

void PySlaveInstance::GetReal(const cppfmu::FMIValueReference* vr, std::size_t nvr, cppfmu::FMIReal* values) const
{
    py_safe_run([this, &vr, nvr, &values](PyGILState_STATE gilState) {
        PyObject* vrs = PyList_New(nvr);
        for (int i = 0; i < nvr; i++) {
            PyList_SetItem(vrs, i, Py_BuildValue("i", vr[i]));
        }

        auto refs = PyObject_CallMethod(pInstance_, "get_real", "O", vrs);
        Py_DECREF(vrs);
        if (refs == nullptr) {
            handle_py_exception("[getReal] PyObject_CallMethod", gilState);
        }

        for (int i = 0; i < nvr; i++) {
            PyObject* value = PyList_GetItem(refs, i);
            values[i] = PyFloat_AsDouble(value);
        }
        Py_DECREF(refs);
        clearLogBuffer();
    });
}

void PySlaveInstance::GetInteger(const cppfmu::FMIValueReference* vr, std::size_t nvr, cppfmu::FMIInteger* values) const
{
    py_safe_run([this, &vr, nvr, &values](PyGILState_STATE gilState) {
        PyObject* vrs = PyList_New(nvr);
        for (int i = 0; i < nvr; i++) {
            PyList_SetItem(vrs, i, Py_BuildValue("i", vr[i]));
        }
        auto refs = PyObject_CallMethod(pInstance_, "get_integer", "O", vrs);
        Py_DECREF(vrs);
        if (refs == nullptr) {
            handle_py_exception("[getInteger] PyObject_CallMethod", gilState);
        }

        for (int i = 0; i < nvr; i++) {
            PyObject* value = PyList_GetItem(refs, i);
            values[i] = static_cast<cppfmu::FMIInteger>(PyLong_AsLong(value));
        }
        Py_DECREF(refs);
        clearLogBuffer();
    });
}

void PySlaveInstance::GetBoolean(const cppfmu::FMIValueReference* vr, std::size_t nvr, cppfmu::FMIBoolean* values) const
{
    py_safe_run([this, &vr, nvr, &values](PyGILState_STATE gilState) {
        PyObject* vrs = PyList_New(nvr);
        for (int i = 0; i < nvr; i++) {
            PyList_SetItem(vrs, i, Py_BuildValue("i", vr[i]));
        }
        auto refs = PyObject_CallMethod(pInstance_, "get_boolean", "O", vrs);
        Py_DECREF(vrs);
        if (refs == nullptr) {
            handle_py_exception("[getBoolean] PyObject_CallMethod", gilState);
        }

        for (int i = 0; i < nvr; i++) {
            PyObject* value = PyList_GetItem(refs, i);
            values[i] = PyObject_IsTrue(value);
        }
        Py_DECREF(refs);
        clearLogBuffer();
    });
}

void PySlaveInstance::GetString(const cppfmu::FMIValueReference* vr, std::size_t nvr, cppfmu::FMIString* values) const
{
    py_safe_run([this, &vr, nvr, &values](PyGILState_STATE gilState) {
        clearStrBuffer();
        PyObject* vrs = PyList_New(nvr);
        for (int i = 0; i < nvr; i++) {
            PyList_SetItem(vrs, i, Py_BuildValue("i", vr[i]));
        }
        auto refs = PyObject_CallMethod(pInstance_, "get_string", "O", vrs);
        Py_DECREF(vrs);
        if (refs == nullptr) {
            handle_py_exception("[getString] PyObject_CallMethod", gilState);
        }

        for (int i = 0; i < nvr; i++) {
            PyObject* value = PyUnicode_AsEncodedString(PyList_GetItem(refs, i), "utf-8", nullptr);
            values[i] = PyBytes_AsString(value);
            strBuffer.emplace_back(value);
        }
        Py_DECREF(refs);
        clearLogBuffer();
    });
}

void PySlaveInstance::GetFMUstate(fmi2FMUstate& state)
{
    py_safe_run([this, &state](PyGILState_STATE gilState) {
        auto f = PyObject_CallMethod(pInstance_, "_get_fmu_state", nullptr);
        if (f == nullptr) {
            handle_py_exception("[_get_fmu_state] PyObject_CallMethod", gilState);
        }
        state = reinterpret_cast<fmi2FMUstate*>(f);
        clearLogBuffer();
    });
}

void PySlaveInstance::SetFMUstate(const fmi2FMUstate& state)
{
    py_safe_run([this, &state](PyGILState_STATE gilState) {
        auto pyState = reinterpret_cast<PyObject*>(state);
        auto f = PyObject_CallMethod(pInstance_, "_set_fmu_state", "(O)", pyState);
        if (f == nullptr) {
            handle_py_exception("[_set_fmu_state] PyObject_CallMethod", gilState);
        }
        clearLogBuffer();
    });
}

void PySlaveInstance::FreeFMUstate(fmi2FMUstate& state)
{
    py_safe_run([this, &state](PyGILState_STATE gilState) {
        auto f = reinterpret_cast<PyObject*>(state);
        Py_XDECREF(f);
    });
}

size_t PySlaveInstance::SerializedFMUstateSize(const fmi2FMUstate& state)
{
    size_t size;
    py_safe_run([this, &state, &size](PyGILState_STATE gilState) {
        auto pyState = reinterpret_cast<PyObject*>(state);
        PyObject* pyStateBytes = PyObject_CallMethod(pClass_, "_fmu_state_to_bytes", "(O)", pyState);
        if (pyStateBytes == nullptr) {
            handle_py_exception("[SerializedFMUstateSize] PyObject_CallMethod", gilState);
        }
        size = PyBytes_Size(pyStateBytes);
        Py_DECREF(pyStateBytes);
        clearLogBuffer();
    });
    return size;
}

void PySlaveInstance::SerializeFMUstate(const fmi2FMUstate& state, fmi2Byte* bytes, size_t size)
{
    py_safe_run([this, &state, &bytes, size](PyGILState_STATE gilState) {
        auto pyState = reinterpret_cast<PyObject*>(state);
        PyObject* pyStateBytes = PyObject_CallMethod(pClass_, "_fmu_state_to_bytes", "(O)", pyState);
        if (pyStateBytes == nullptr) {
            handle_py_exception("[SerializeFMUstate] PyObject_CallMethod", gilState);
        }
        char* c = PyBytes_AsString(pyStateBytes);
        if (c == nullptr) {
            handle_py_exception("[SerializeFMUstate] PyBytes_AsString", gilState);
        }
        for (int i = 0; i < size; i++) {
            bytes[i] = c[i];
        }
        Py_DECREF(pyStateBytes);
        clearLogBuffer();
    });
}

void PySlaveInstance::DeSerializeFMUstate(const fmi2Byte bytes[], size_t size, fmi2FMUstate& state)
{
    py_safe_run([this, &bytes, size, &state](PyGILState_STATE gilState) {
        PyObject* pyStateBytes = PyBytes_FromStringAndSize(bytes, size);
        if (pyStateBytes == nullptr) {
            handle_py_exception("[DeSerializeFMUstate] PyBytes_FromStringAndSize", gilState);
        }
        PyObject* pyState = PyObject_CallMethod(pClass_, "_fmu_state_from_bytes", "(O)", pyStateBytes);
        if (pyState == nullptr) {
            handle_py_exception("[DeSerializeFMUstate] PyObject_CallMethod", gilState);
        }
        state = reinterpret_cast<fmi2FMUstate*>(pyState);
        Py_DECREF(pyStateBytes);
        clearLogBuffer();
    });
}

void PySlaveInstance::handle_py_exception(const std::string& what, PyGILState_STATE gilState) const
{
    auto err = PyErr_Occurred();
    if (err != nullptr) {
        cleanPyObject();

        PyObject *pExcType, *pExcValue, *pExcTraceback;
        PyErr_Fetch(&pExcType, &pExcValue, &pExcTraceback);

        std::ostringstream oss;
        oss << "Fatal py exception encountered: ";
        oss << what << "\n";
        if (pExcValue != nullptr) {
            PyObject* pRepr = PyObject_Repr(pExcValue);
            PyObject* pyStr = PyUnicode_AsEncodedString(pRepr, "utf-8", nullptr);
            oss << PyBytes_AsString(pyStr);
            Py_DECREF(pyStr);
            Py_DECREF(pRepr);
        } else {
            oss << "unknown error";
        }

        PyErr_Clear();

        Py_XDECREF(pExcType);
        Py_XDECREF(pExcValue);
        Py_XDECREF(pExcTraceback);

        PyGILState_Release(gilState);

        auto msg = oss.str();
        throw cppfmu::FatalError(msg.c_str());
    }
}

PySlaveInstance::~PySlaveInstance()
{
    py_safe_run([this](PyGILState_STATE gilState) {
        cleanPyObject();
    });
}

} // namespace pythonfmu

std::unique_ptr<pythonfmu::PyState> pyState = nullptr;

cppfmu::UniquePtr<cppfmu::SlaveInstance> CppfmuInstantiateSlave(
    cppfmu::FMIString instanceName,
    cppfmu::FMIString,
    cppfmu::FMIString fmuResourceLocation,
    cppfmu::FMIString,
    cppfmu::FMIReal,
    cppfmu::FMIBoolean visible,
    cppfmu::FMIBoolean,
    cppfmu::Memory memory,
    const cppfmu::Logger& logger)
{

    auto resources = std::string(fmuResourceLocation);
    auto find = resources.find("file://");

    if (find != std::string::npos) {
#ifdef _MSC_VER
        resources.replace(find, 8, "");
#else
        resources.replace(find, 7, "");
#endif
    }

    if (pyState == nullptr) {
        pyState = std::make_unique<pythonfmu::PyState>();
    }

    return cppfmu::AllocateUnique<pythonfmu::PySlaveInstance>(
        memory, instanceName, resources, logger, visible);
}

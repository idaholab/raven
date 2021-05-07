
#ifndef PYTHONFMU_SLAVEINSTANCE_HPP
#define PYTHONFMU_SLAVEINSTANCE_HPP

#include "cppfmu/cppfmu_cs.hpp"

#include <Python.h>
#include <string>
#include <vector>

namespace pythonfmu
{

class PySlaveInstance : public cppfmu::SlaveInstance
{

public:
    PySlaveInstance(std::string instanceName, std::string resources, const cppfmu::Logger& logger, bool visible);

    void initialize(PyGILState_STATE gilState);

    void SetupExperiment(cppfmu::FMIBoolean toleranceDefined, cppfmu::FMIReal tolerance, cppfmu::FMIReal tStart, cppfmu::FMIBoolean stopTimeDefined, cppfmu::FMIReal tStop) override;
    void EnterInitializationMode() override;
    void ExitInitializationMode() override;
    void Terminate() override;
    void Reset() override;
    bool DoStep(cppfmu::FMIReal currentCommunicationPoint, cppfmu::FMIReal communicationStepSize, cppfmu::FMIBoolean newStep, cppfmu::FMIReal& endOfStep) override;

    void SetReal(const cppfmu::FMIValueReference* vr, std::size_t nvr, const cppfmu::FMIReal* value) override;
    void SetInteger(const cppfmu::FMIValueReference* vr, std::size_t nvr, const cppfmu::FMIInteger* value) override;
    void SetBoolean(const cppfmu::FMIValueReference* vr, std::size_t nvr, const cppfmu::FMIBoolean* value) override;
    void SetString(const cppfmu::FMIValueReference* vr, std::size_t nvr, cppfmu::FMIString const* value) override;

    void GetReal(const cppfmu::FMIValueReference* vr, std::size_t nvr, cppfmu::FMIReal* value) const override;
    void GetInteger(const cppfmu::FMIValueReference* vr, std::size_t nvr, cppfmu::FMIInteger* value) const override;
    void GetBoolean(const cppfmu::FMIValueReference* vr, std::size_t nvr, cppfmu::FMIBoolean* value) const override;
    void GetString(const cppfmu::FMIValueReference* vr, std::size_t nvr, cppfmu::FMIString* value) const override;

    void GetFMUstate(fmi2FMUstate& state) override;
    void SetFMUstate(const fmi2FMUstate& state) override;
    void FreeFMUstate(fmi2FMUstate& state) override;

    size_t SerializedFMUstateSize(const fmi2FMUstate& state) override;
    void SerializeFMUstate(const fmi2FMUstate& state, fmi2Byte bytes[], size_t size) override;
    void DeSerializeFMUstate(const fmi2Byte bytes[], size_t size, fmi2FMUstate& state) override;

    void clearLogBuffer() const;

    ~PySlaveInstance() override;

private:
    PyObject* pClass_;
    PyObject* pInstance_{};
    PyObject* pMessages_{};

    const bool visible_;
    const std::string instanceName_;
    const std::string resources_;
    const cppfmu::Logger& logger_;

    mutable std::vector<PyObject*> strBuffer;
    mutable std::vector<PyObject*> logStrBuffer;

    void handle_py_exception(const std::string& what, PyGILState_STATE gilState) const;

    inline void clearStrBuffer() const
    {
        if (!strBuffer.empty()) {
            for (auto obj : strBuffer) {
                Py_DECREF(obj);
            }
            strBuffer.clear();
        }
    }

    inline void clearLogStrBuffer() const
    {
        if (!logStrBuffer.empty()) {
            for (auto obj : logStrBuffer) {
                Py_DECREF(obj);
            }
            logStrBuffer.clear();
        }
    }

    inline void cleanPyObject() const
    {
        clearLogBuffer();
        clearLogStrBuffer();
        clearStrBuffer();
        Py_XDECREF(pClass_);
        Py_XDECREF(pInstance_);
        Py_XDECREF(pMessages_);
    }
};

} // namespace pythonfmu

#endif

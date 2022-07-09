/* Copyright 2016-2019, SINTEF Ocean.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "cppfmu/cppfmu_cs.hpp"

#include <stdexcept>


namespace cppfmu
{

// =============================================================================
// SlaveInstance
// =============================================================================


void SlaveInstance::SetupExperiment(
    FMIBoolean /*toleranceDefined*/,
    FMIReal /*tolerance*/,
    FMIReal /*tStart*/,
    FMIBoolean /*stopTimeDefined*/,
    FMIReal /*tStop*/)
{
    // Do nothing
}


void SlaveInstance::EnterInitializationMode()
{
    // Do nothing
}


void SlaveInstance::ExitInitializationMode()
{
    // Do nothing
}


void SlaveInstance::Terminate()
{
    // Do nothing
}


void SlaveInstance::Reset()
{
    // Do nothing
}


void SlaveInstance::SetReal(
    const FMIValueReference /*vr*/[],
    std::size_t nvr,
    const FMIReal /*value*/[])
{
    if (nvr != 0) {
        throw std::logic_error("Attempted to set nonexistent variable");
    }
}


void SlaveInstance::SetInteger(
    const FMIValueReference /*vr*/[],
    std::size_t nvr,
    const FMIInteger /*value*/[])
{
    if (nvr != 0) {
        throw std::logic_error("Attempted to set nonexistent variable");
    }
}


void SlaveInstance::SetBoolean(
    const FMIValueReference /*vr*/[],
    std::size_t nvr,
    const FMIBoolean /*value*/[])
{
    if (nvr != 0) {
        throw std::logic_error("Attempted to set nonexistent variable");
    }
}


void SlaveInstance::SetString(
    const FMIValueReference /*vr*/[],
    std::size_t nvr,
    const FMIString /*value*/[])
{
    if (nvr != 0) {
        throw std::logic_error("Attempted to set nonexistent variable");
    }
}


void SlaveInstance::GetReal(
    const FMIValueReference /*vr*/[],
    std::size_t nvr,
    FMIReal /*value*/[]) const
{
    if (nvr != 0) {
        throw std::logic_error("Attempted to get nonexistent variable");
    }
}


void SlaveInstance::GetInteger(
    const FMIValueReference /*vr*/[],
    std::size_t nvr,
    FMIInteger /*value*/[]) const
{
    if (nvr != 0) {
        throw std::logic_error("Attempted to get nonexistent variable");
    }
}


void SlaveInstance::GetBoolean(
    const FMIValueReference /*vr*/[],
    std::size_t nvr,
    FMIBoolean /*value*/[]) const
{
    if (nvr != 0) {
        throw std::logic_error("Attempted to set nonexistent variable");
    }
}


void SlaveInstance::GetString(
    const FMIValueReference /*vr*/[],
    std::size_t nvr,
    FMIString /*value*/[]) const
{
    if (nvr != 0) {
        throw std::logic_error("Attempted to set nonexistent variable");
    }
}


SlaveInstance::~SlaveInstance() CPPFMU_NOEXCEPT
{
    // Do nothing
}


} // namespace cppfmu

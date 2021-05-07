/* Copyright 2016-2019, SINTEF Ocean.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CPPFMU_COMMON_HPP
#define CPPFMU_COMMON_HPP

#include <algorithm>    // std::find()
#include <cstddef>      // std::size_t
#include <functional>   // std::function
#include <memory>       // std::shared_ptr, std::unique_ptr
#include <new>          // std::bad_alloc
#include <stdexcept>    // std::runtime_error
#include <string>       // std::basic_string, std::char_traits
#include <utility>      // std::forward
#include <vector>

extern "C"
{
#ifdef CPPFMU_USE_FMI_1_0
#   include "fmiFunctions.h"
#else
#    include "fmi/fmi2Functions.h"
#endif
}


// CPPFMU_NOEXCEPT evaluates to 'noexcept' on compilers that support it.
#if (__cplusplus >= 201103L) || (defined(_MSC_VER) && _MSC_VER >= 1900)
#   define CPPFMU_NOEXCEPT noexcept
#else
#   define CPPFMU_NOEXCEPT
#endif


namespace cppfmu
{

// Aliases for FMI types and enums
#ifdef CPPFMU_USE_FMI_1_0
    typedef fmiReal FMIReal;
    typedef fmiInteger FMIInteger;
    typedef fmiBoolean FMIBoolean;
    typedef fmiString FMIString;
    typedef fmiCallbackFunctions FMICallbackFunctions;
    typedef fmiCallbackAllocateMemory FMICallbackAllocateMemory;
    typedef fmiCallbackFreeMemory FMICallbackFreeMemory;
    typedef fmiCallbackLogger FMICallbackLogger;
    typedef fmiComponent FMIComponent;
    typedef fmiComponent FMIComponentEnvironment;
    typedef fmiStatus FMIStatus;
    typedef fmiValueReference FMIValueReference;

    const FMIBoolean FMIFalse = fmiFalse;
    const FMIBoolean FMITrue = fmiTrue;

    const FMIStatus FMIOK = fmiOK;
    const FMIStatus FMIWarning = fmiWarning;
    const FMIStatus FMIDiscard = fmiDiscard;
    const FMIStatus FMIError = fmiError;
    const FMIStatus FMIFatal = fmiFatal;
    const FMIStatus FMIPending = fmiPending;
#else
    typedef fmi2Real FMIReal;
    typedef fmi2Integer FMIInteger;
    typedef fmi2Boolean FMIBoolean;
    typedef fmi2String FMIString;
    typedef fmi2CallbackFunctions FMICallbackFunctions;
    typedef fmi2CallbackAllocateMemory FMICallbackAllocateMemory;
    typedef fmi2CallbackFreeMemory FMICallbackFreeMemory;
    typedef fmi2CallbackLogger FMICallbackLogger;
    typedef fmi2Component FMIComponent;
    typedef fmi2ComponentEnvironment FMIComponentEnvironment;
    typedef fmi2Status FMIStatus;
    typedef fmi2ValueReference FMIValueReference;

    const FMIBoolean FMIFalse = fmi2False;
    const FMIBoolean FMITrue = fmi2True;

    const FMIStatus FMIOK = fmi2OK;
    const FMIStatus FMIWarning = fmi2Warning;
    const FMIStatus FMIDiscard = fmi2Discard;
    const FMIStatus FMIError = fmi2Error;
    const FMIStatus FMIFatal = fmi2Fatal;
    const FMIStatus FMIPending = fmi2Pending;
#endif


// ============================================================================
// ERROR HANDLING
// ============================================================================


/* Exception class that signals "fatal error", i.e. an error which means that
 * not only is the current model instance invalid, but all other instances of
 * the same model too.
 */
class FatalError : public std::runtime_error
{
public:
    FatalError(const char* msg) CPPFMU_NOEXCEPT : std::runtime_error{msg} { }
};


// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================


/* A wrapper class for the FMI memory allocation and deallocation functions.
 * Alloc() and Free() simply forward to the functions provided by the
 * simulation environment.
 */
class Memory
{
public:
    explicit Memory(const FMICallbackFunctions& callbackFunctions)
        : m_alloc{callbackFunctions.allocateMemory}
        , m_free{callbackFunctions.freeMemory}
    {
    }

    // Allocates memory for 'nObj' objects of size 'size'.
    void* Alloc(std::size_t nObj, std::size_t size) CPPFMU_NOEXCEPT
    {
        return m_alloc(nObj, size);
    }

    // Frees the memory pointed to by 'ptr'.
    void Free(void* ptr) CPPFMU_NOEXCEPT
    {
        m_free(ptr);
    }

    bool operator==(const Memory& rhs) const CPPFMU_NOEXCEPT
    {
        return m_alloc == rhs.m_alloc && m_free == rhs.m_free;
    }

    bool operator!=(const Memory& rhs) const CPPFMU_NOEXCEPT
    {
        return !operator==(rhs);
    }

private:
    FMICallbackAllocateMemory m_alloc;
    FMICallbackFreeMemory m_free;
};


/* A class that satisfies the Allocator concept, and which can therefore be
 * used to manage memory for the standard C++ containers.
 *
 * For information about the various member functions, we refer to reference
 * material for the Allocator concept, e.g.:
 * http://en.cppreference.com/w/cpp/concept/Allocator
 */
template<typename T>
class Allocator
{
public:
    using value_type = T;

    explicit Allocator(const Memory& memory) : m_memory{memory} { }

    template<typename U>
    Allocator(const Allocator<U>& other) CPPFMU_NOEXCEPT
        : m_memory{other.m_memory}
    {
    }

    T* allocate(std::size_t n)
    {
        if (n == 0) return nullptr;
        if (auto m = m_memory.Alloc(n, sizeof(T))) {
            return reinterpret_cast<T*>(m);
        } else {
            throw std::bad_alloc();
        }
    }

    void deallocate(T* p, std::size_t n) CPPFMU_NOEXCEPT
    {
        if (n > 0) {
            m_memory.Free(p);
        }
    }

    bool operator==(const Allocator& rhs) const CPPFMU_NOEXCEPT
    {
        return m_memory == rhs.m_memory;
    }

    bool operator!=(const Allocator& rhs) const CPPFMU_NOEXCEPT
    {
        return !operator==(rhs);
    }

    // -------------------------------------------------------------------------
    // None of the following are required by C++11, yet they are, variously,
    // required by GCC and MSVC.

    template<typename U>
    struct rebind { using other = Allocator<U>; };

#if defined(__GNUC__) && (__GNUC__ < 5)
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
    Allocator() : m_memory{FMICallbackFunctions{}} { }
#pragma GCC diagnostic pop
#endif

#ifdef _MSC_VER
    template<typename U, typename... Args>
    void construct(U* p, Args&&... args)
    {
        ::new((void*) p) U(std::forward<Args>(args)...);
    }

    template<typename U>
    void destroy(U* p)
    {
        p->~U();
    }
#endif

    // -------------------------------------------------------------------------

private:
    template<typename U>
    friend class Allocator;

    Memory m_memory;
};


// An alias for a string type that uses cppfmu::Allocator to manage memory.
using String = std::basic_string<char, std::char_traits<char>, Allocator<char>>;


// Returns a String whose contents are equal to those of 'string'.
inline String CopyString(const Memory& memory, FMIString string)
{
    return String{string, Allocator<char>{memory}};
}


/* Allocates memory for a single object of type T and runs its constructor,
 * in the style of the built-in 'new' operator.  Any arguments in 'args'
 * are forwarded to the constructor.
 */
template<typename T, typename... Args>
T* New(const Memory& memory, Args&&... args)
{
    auto alloc = Allocator<T>{memory};
    const auto ptr = std::allocator_traits<decltype(alloc)>::allocate(alloc, 1);
    try {
        std::allocator_traits<decltype(alloc)>::construct(
            alloc,
            ptr,
            std::forward<Args>(args)...);
    } catch (...) {
        std::allocator_traits<decltype(alloc)>::deallocate(alloc, ptr, 1);
        throw;
    }
    return ptr;
}


/* Destroys and deallocates memory for an object of type T, in the style of
 * the built-in 'delete' operator.
 */
template<typename T>
void Delete(const Memory& memory, T* obj) CPPFMU_NOEXCEPT
{
    auto alloc = Allocator<T>{memory};
    std::allocator_traits<decltype(alloc)>::destroy(alloc, obj);
    std::allocator_traits<decltype(alloc)>::deallocate(alloc, obj, 1);
}


/* An alias for a std::unique_ptr specialisation where the deleter is general
 * and independent of the type of the object pointed to.  This is used for the
 * return type of AllocateUnique() below.
 */
template<typename T>
using UniquePtr = std::unique_ptr<T, std::function<void(void*)>>;


/* Creates an object of type T which is managed by a std::unique_ptr.
 * The object is created using cppfmu::New(), and when the time comes, it is
 * destroyed using cppfmu::Delete().
 */
template<typename T, typename... Args>
UniquePtr<T> AllocateUnique(const Memory& memory, Args&&... args)
{
    return UniquePtr<T>{
        New<T>(memory, std::forward<Args>(args)...),
        [memory] (void* ptr) { Delete(memory, reinterpret_cast<T*>(ptr)); }};
}


// ============================================================================
// LOGGING
// ============================================================================

namespace detail
{
    template<typename Container, typename Item>
    bool CanFind(const Container& container, const Item& item)
    {
        return container.end() != std::find(
            container.begin(),
            container.end(),
            item);
    }
}


/* A class that can be used to log messages from model code.  All messages are
 * forwarded to the logging facilities provided by the simulation environment.
 */
class Logger
{
public:
    struct Settings
    {
        Settings(const Memory& memory)
            : loggedCategories(Allocator<String>{memory})
        { }

        bool debugLoggingEnabled = false;
        std::vector<String, Allocator<String>> loggedCategories;
    };

    Logger(
        FMIComponentEnvironment component,
        String instanceName,
        FMICallbackFunctions callbackFunctions,
        std::shared_ptr<Settings> settings)
        : m_component{component}
        , m_instanceName(std::move(instanceName))
        , m_fmiLogger{callbackFunctions.logger}
        , m_settings{settings}
    {
    }

    // Logs a message.
    template<typename... Args>
    void Log(
        FMIStatus status,
        FMIString category,
        FMIString message,
        Args&&... args) CPPFMU_NOEXCEPT
    {
        if (m_settings->loggedCategories.empty() ||
            detail::CanFind(m_settings->loggedCategories, category)) {
            m_fmiLogger(
                m_component,
                m_instanceName.c_str(),
                status,
                category,
                message,
                std::forward<Args>(args)...);
        }
    }

    /* Logs a debug message (if debug logging is enabled by the simulation
     * environment).
     */
    template<typename... Args>
    void DebugLog(
        FMIStatus status,
        FMIString category,
        FMIString message,
        Args&&... args) CPPFMU_NOEXCEPT
    {
        if (m_settings->debugLoggingEnabled) {
            Log(
                status,
                category,
                message,
                std::forward<Args>(args)...);
        }
    }

private:
    const FMIComponentEnvironment m_component;
    const String m_instanceName;
    const FMICallbackLogger m_fmiLogger;
    std::shared_ptr<Settings> m_settings;
};


} // namespace cppfmu
#endif // header guard

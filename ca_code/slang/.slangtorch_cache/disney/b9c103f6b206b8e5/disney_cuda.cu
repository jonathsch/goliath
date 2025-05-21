#define SLANG_PRELUDE_EXPORT

#ifdef __CUDACC_RTC__
#define SLANG_CUDA_RTC 1
#else
#define SLANG_CUDA_RTC 0
#endif

#if SLANG_CUDA_RTC

#else

#include <cstdint>
#include <stdio.h>

#endif

// Define SLANG_CUDA_ENABLE_HALF to use the cuda_fp16 include to add half support.
// For this to work NVRTC needs to have the path to the CUDA SDK.
//
// As it stands the includes paths defined for Slang are passed down to NVRTC. Similarly defines
// defined for the Slang compile are passed down.

#ifdef SLANG_CUDA_ENABLE_HALF
// We don't want half2 operators, because it will implement comparison operators that return a
// bool(!). We want to generate those functions. Doing so means that we will have to define all
// the other half2 operators.
#define __CUDA_NO_HALF2_OPERATORS__
#include <cuda_fp16.h>
#endif

#ifdef SLANG_CUDA_ENABLE_OPTIX
#include <optix.h>
#endif

// Define slang offsetof implementation
#ifndef SLANG_OFFSET_OF
#define SLANG_OFFSET_OF(type, member) (size_t)((char*)&(((type*)0)->member) - (char*)0)
#endif

#ifndef SLANG_ALIGN_OF
#define SLANG_ALIGN_OF(type) __alignof__(type)
#endif

// Must be large enough to cause overflow and therefore infinity
#ifndef SLANG_INFINITY
#define SLANG_INFINITY ((float)(1e+300 * 1e+300))
#endif

// For now we'll disable any asserts in this prelude
#define SLANG_PRELUDE_ASSERT(x)

#ifndef SLANG_CUDA_WARP_SIZE
#define SLANG_CUDA_WARP_SIZE 32
#endif

#define SLANG_CUDA_WARP_MASK \
    (SLANG_CUDA_WARP_SIZE - 1) // Used for masking threadIdx.x to the warp lane index
#define SLANG_CUDA_WARP_BITMASK (~int(0))

//
#define SLANG_FORCE_INLINE inline

#define SLANG_CUDA_CALL __device__

#define SLANG_FORCE_INLINE inline
#define SLANG_INLINE inline


// Since we are using unsigned arithmatic care is need in this comparison.
// It is *assumed* that sizeInBytes >= elemSize. Which means (sizeInBytes >= elemSize) >= 0
// Which means only a single test is needed

// Asserts for bounds checking.
// It is assumed index/count are unsigned types.
#define SLANG_BOUND_ASSERT(index, count) SLANG_PRELUDE_ASSERT(index < count);
#define SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_PRELUDE_ASSERT(index <= (sizeInBytes - elemSize) && (index & 3) == 0);

// Macros to zero index if an access is out of range
#define SLANG_BOUND_ZERO_INDEX(index, count) index = (index < count) ? index : 0;
#define SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    index = (index <= (sizeInBytes - elemSize)) ? index : 0;

// The 'FIX' macro define how the index is fixed. The default is to do nothing. If
// SLANG_ENABLE_BOUND_ZERO_INDEX the fix macro will zero the index, if out of range
#ifdef SLANG_ENABLE_BOUND_ZERO_INDEX
#define SLANG_BOUND_FIX(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#define SLANG_BOUND_FIX_FIXED_ARRAY(index, count) \
    SLANG_BOUND_ZERO_INDEX(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#else
#define SLANG_BOUND_FIX(index, count)
#define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#define SLANG_BOUND_FIX_FIXED_ARRAY(index, count)
#endif

#ifndef SLANG_BOUND_CHECK
#define SLANG_BOUND_CHECK(index, count) \
    SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX(index, count)
#endif

#ifndef SLANG_BOUND_CHECK_BYTE_ADDRESS
#define SLANG_BOUND_CHECK_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes)    \
    SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#endif

#ifndef SLANG_BOUND_CHECK_FIXED_ARRAY
#define SLANG_BOUND_CHECK_FIXED_ARRAY(index, count) \
    SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX_FIXED_ARRAY(index, count)
#endif

// This macro handles how out-of-range surface coordinates are handled;
// I can equal
// cudaBoundaryModeClamp, in which case out-of-range coordinates are clamped to the valid range
// cudaBoundaryModeZero, in which case out-of-range reads return zero and out-of-range writes are
// ignored cudaBoundaryModeTrap, in which case out-of-range accesses cause the kernel execution to
// fail.

#ifndef SLANG_CUDA_BOUNDARY_MODE
#define SLANG_CUDA_BOUNDARY_MODE cudaBoundaryModeZero

// Can be one of SLANG_CUDA_PTX_BOUNDARY_MODE. Only applies *PTX* emitted CUDA operations
// which currently is just RWTextureRW format writes
//
// .trap         causes an execution trap on out-of-bounds addresses
// .clamp        stores data at the nearest surface location (sized appropriately)
// .zero         drops stores to out-of-bounds addresses

#define SLANG_PTX_BOUNDARY_MODE "zero"
#endif

struct TypeInfo
{
    size_t typeSize;
};

template<typename T, size_t SIZE>
struct FixedArray
{
    SLANG_CUDA_CALL const T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE);
        return m_data[index];
    }
    SLANG_CUDA_CALL T& operator[](size_t index)
    {
        SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE);
        return m_data[index];
    }

    T m_data[SIZE];
};

// An array that has no specified size, becomes a 'Array'. This stores the size so it can
// potentially do bounds checking.
template<typename T>
struct Array
{
    SLANG_CUDA_CALL const T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    SLANG_CUDA_CALL T& operator[](size_t index)
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }

    T* data;
    size_t count;
};

// Typically defined in cuda.h, but we can't ship/rely on that, so just define here
typedef unsigned long long CUtexObject;
typedef unsigned long long CUsurfObject;

// On CUDA sampler state is actually bound up with the texture object. We have a SamplerState type,
// backed as a pointer, to simplify code generation, with the downside that such a binding will take
// up uniform space, even though it will have no effect.
// TODO(JS): Consider ways to strip use of variables of this type so have no binding,
struct SamplerStateUnused;
typedef SamplerStateUnused* SamplerState;


// TODO(JS): Not clear yet if this can be handled on CUDA, by just ignoring.
// For now, just map to the index type.
typedef size_t NonUniformResourceIndex;

// Code generator will generate the specific type
template<typename T, int ROWS, int COLS>
struct Matrix;

typedef int1 bool1;
typedef int2 bool2;
typedef int3 bool3;
typedef int4 bool4;

#if SLANG_CUDA_RTC

typedef signed char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long long int64_t;

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#endif

typedef long long longlong;
typedef unsigned long long ulonglong;

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;

union Union32
{
    uint32_t u;
    int32_t i;
    float f;
};

union Union64
{
    uint64_t u;
    int64_t i;
    double d;
};

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL float make_float(T val)
{
    return (float)val;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float _slang_fmod(float x, float y)
{
    return ::fmodf(x, y);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double _slang_fmod(double x, double y)
{
    return ::fmod(x, y);
}

#if SLANG_CUDA_ENABLE_HALF

// Add the other vector half types
struct __half1
{
    __half x;
};
struct __align__(4) __half3
{
    __half x, y, z;
};
struct __align__(4) __half4
{
    __half x, y, z, w;
};
#endif

#define SLANG_VECTOR_GET_ELEMENT(T)                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##1 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##2 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##3 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##4 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }
SLANG_VECTOR_GET_ELEMENT(int)
SLANG_VECTOR_GET_ELEMENT(uint)
SLANG_VECTOR_GET_ELEMENT(short)
SLANG_VECTOR_GET_ELEMENT(ushort)
SLANG_VECTOR_GET_ELEMENT(char)
SLANG_VECTOR_GET_ELEMENT(uchar)
SLANG_VECTOR_GET_ELEMENT(longlong)
SLANG_VECTOR_GET_ELEMENT(ulonglong)
SLANG_VECTOR_GET_ELEMENT(float)
SLANG_VECTOR_GET_ELEMENT(double)

#define SLANG_VECTOR_GET_ELEMENT_PTR(T)                                                      \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(T##1 * x, int index) \
    {                                                                                        \
        return ((T*)(x)) + index;                                                            \
    }                                                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(T##2 * x, int index) \
    {                                                                                        \
        return ((T*)(x)) + index;                                                            \
    }                                                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(T##3 * x, int index) \
    {                                                                                        \
        return ((T*)(x)) + index;                                                            \
    }                                                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(T##4 * x, int index) \
    {                                                                                        \
        return ((T*)(x)) + index;                                                            \
    }
SLANG_VECTOR_GET_ELEMENT_PTR(int)
SLANG_VECTOR_GET_ELEMENT_PTR(uint)
SLANG_VECTOR_GET_ELEMENT_PTR(short)
SLANG_VECTOR_GET_ELEMENT_PTR(ushort)
SLANG_VECTOR_GET_ELEMENT_PTR(char)
SLANG_VECTOR_GET_ELEMENT_PTR(uchar)
SLANG_VECTOR_GET_ELEMENT_PTR(longlong)
SLANG_VECTOR_GET_ELEMENT_PTR(ulonglong)
SLANG_VECTOR_GET_ELEMENT_PTR(float)
SLANG_VECTOR_GET_ELEMENT_PTR(double)

#if SLANG_CUDA_ENABLE_HALF
SLANG_VECTOR_GET_ELEMENT(__half)
SLANG_VECTOR_GET_ELEMENT_PTR(__half)
#endif

#define SLANG_CUDA_VECTOR_BINARY_OP(T, n, op)                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n operator op(T##n thisVal, T##n other)             \
    {                                                                                         \
        T##n result;                                                                          \
        for (int i = 0; i < n; i++)                                                           \
            *_slang_vector_get_element_ptr(&result, i) =                                      \
                _slang_vector_get_element(thisVal, i) op _slang_vector_get_element(other, i); \
        return result;                                                                        \
    }
#define SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, op)                                \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL bool##n operator op(T##n thisVal, T##n other) \
    {                                                                                \
        bool##n result;                                                              \
        for (int i = 0; i < n; i++)                                                  \
            *_slang_vector_get_element_ptr(&result, i) =                             \
                (int)(_slang_vector_get_element(thisVal, i)                          \
                          op _slang_vector_get_element(other, i));                   \
        return result;                                                               \
    }
#define SLANG_CUDA_VECTOR_UNARY_OP(T, n, op)                                                       \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n operator op(T##n thisVal)                              \
    {                                                                                              \
        T##n result;                                                                               \
        for (int i = 0; i < n; i++)                                                                \
            *_slang_vector_get_element_ptr(&result, i) = op _slang_vector_get_element(thisVal, i); \
        return result;                                                                             \
    }

#define SLANG_CUDA_VECTOR_INT_OP(T, n)            \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, +)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, -)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, *)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, /)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, %)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, ^)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, &)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, |)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, &&)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, ||)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, >>)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, <<)         \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, ==) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, !=) \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, !)           \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, -)           \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, ~)

#define SLANG_CUDA_VECTOR_INT_OPS(T) \
    SLANG_CUDA_VECTOR_INT_OP(T, 2)   \
    SLANG_CUDA_VECTOR_INT_OP(T, 3)   \
    SLANG_CUDA_VECTOR_INT_OP(T, 4)

SLANG_CUDA_VECTOR_INT_OPS(int)
SLANG_CUDA_VECTOR_INT_OPS(uint)
SLANG_CUDA_VECTOR_INT_OPS(ushort)
SLANG_CUDA_VECTOR_INT_OPS(short)
SLANG_CUDA_VECTOR_INT_OPS(char)
SLANG_CUDA_VECTOR_INT_OPS(uchar)
SLANG_CUDA_VECTOR_INT_OPS(longlong)
SLANG_CUDA_VECTOR_INT_OPS(ulonglong)

#define SLANG_CUDA_VECTOR_FLOAT_OP(T, n)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, +)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, -)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, *)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, /)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, &&)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, ||)         \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, ==) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, !=) \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, -)
#define SLANG_CUDA_VECTOR_FLOAT_OPS(T) \
    SLANG_CUDA_VECTOR_FLOAT_OP(T, 2)   \
    SLANG_CUDA_VECTOR_FLOAT_OP(T, 3)   \
    SLANG_CUDA_VECTOR_FLOAT_OP(T, 4)

SLANG_CUDA_VECTOR_FLOAT_OPS(float)
SLANG_CUDA_VECTOR_FLOAT_OPS(double)
#if SLANG_CUDA_ENABLE_HALF
SLANG_CUDA_VECTOR_FLOAT_OPS(__half)
#endif
#define SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, n)                                             \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n operator%(const T##n& left, const T##n& right) \
    {                                                                                      \
        T##n result;                                                                       \
        for (int i = 0; i < n; i++)                                                        \
            *_slang_vector_get_element_ptr(&result, i) = _slang_fmod(                      \
                _slang_vector_get_element(left, i),                                        \
                _slang_vector_get_element(right, i));                                      \
        return result;                                                                     \
    }
#define SLANG_CUDA_FLOAT_VECTOR_MOD(T)     \
    SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, 2) \
    SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, 3) \
    SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, 4)

SLANG_CUDA_FLOAT_VECTOR_MOD(float)
SLANG_CUDA_FLOAT_VECTOR_MOD(double)

#if SLANG_CUDA_RTC || SLANG_CUDA_ENABLE_HALF
#define SLANG_MAKE_VECTOR(T)                                                \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##2 make_##T##2(T x, T y)           \
    {                                                                       \
        return T##2 {x, y};                                                 \
    }                                                                       \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##3 make_##T##3(T x, T y, T z)      \
    {                                                                       \
        return T##3 {x, y, z};                                              \
    }                                                                       \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##4 make_##T##4(T x, T y, T z, T w) \
    {                                                                       \
        return T##4 {x, y, z, w};                                           \
    }
#endif

#if SLANG_CUDA_RTC
SLANG_MAKE_VECTOR(int)
SLANG_MAKE_VECTOR(uint)
SLANG_MAKE_VECTOR(short)
SLANG_MAKE_VECTOR(ushort)
SLANG_MAKE_VECTOR(char)
SLANG_MAKE_VECTOR(uchar)
SLANG_MAKE_VECTOR(float)
SLANG_MAKE_VECTOR(double)
SLANG_MAKE_VECTOR(longlong)
SLANG_MAKE_VECTOR(ulonglong)
#endif

#if SLANG_CUDA_ENABLE_HALF
SLANG_MAKE_VECTOR(__half)
#endif

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool1 make_bool1(bool x)
{
    return bool1{x};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool2 make_bool2(bool x, bool y)
{
    return bool2{x, y};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool3 make_bool3(bool x, bool y, bool z)
{
    return bool3{x, y, z};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool4 make_bool4(bool x, bool y, bool z, bool w)
{
    return bool4{x, y, z, w};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool2 make_bool2(bool x)
{
    return bool2{x, x};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool3 make_bool3(bool x)
{
    return bool3{x, x, x};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool4 make_bool4(bool x)
{
    return bool4{x, x, x, x};
}

#if SLANG_CUDA_RTC
#define SLANG_MAKE_VECTOR_FROM_SCALAR(T)                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##1 make_##T##1(T x) \
    {                                                        \
        return T##1 {x};                                     \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##2 make_##T##2(T x) \
    {                                                        \
        return make_##T##2(x, x);                            \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##3 make_##T##3(T x) \
    {                                                        \
        return make_##T##3(x, x, x);                         \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##4 make_##T##4(T x) \
    {                                                        \
        return make_##T##4(x, x, x, x);                      \
    }
#else
#define SLANG_MAKE_VECTOR_FROM_SCALAR(T)                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##2 make_##T##2(T x) \
    {                                                        \
        return make_##T##2(x, x);                            \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##3 make_##T##3(T x) \
    {                                                        \
        return make_##T##3(x, x, x);                         \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##4 make_##T##4(T x) \
    {                                                        \
        return make_##T##4(x, x, x, x);                      \
    }
#endif
SLANG_MAKE_VECTOR_FROM_SCALAR(int)
SLANG_MAKE_VECTOR_FROM_SCALAR(uint)
SLANG_MAKE_VECTOR_FROM_SCALAR(short)
SLANG_MAKE_VECTOR_FROM_SCALAR(ushort)
SLANG_MAKE_VECTOR_FROM_SCALAR(char)
SLANG_MAKE_VECTOR_FROM_SCALAR(uchar)
SLANG_MAKE_VECTOR_FROM_SCALAR(longlong)
SLANG_MAKE_VECTOR_FROM_SCALAR(ulonglong)
SLANG_MAKE_VECTOR_FROM_SCALAR(float)
SLANG_MAKE_VECTOR_FROM_SCALAR(double)
#if SLANG_CUDA_ENABLE_HALF
SLANG_MAKE_VECTOR_FROM_SCALAR(__half)
#if !SLANG_CUDA_RTC
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half1 make___half1(__half x)
{
    return __half1{x};
}
#endif
#endif

#define SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(Fn, T, N)                                            \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##N Fn(T##N* address, T##N val)                           \
    {                                                                                             \
        T##N result;                                                                              \
        for (int i = 0; i < N; i++)                                                               \
            *_slang_vector_get_element_ptr(&result, i) =                                          \
                Fn(_slang_vector_get_element_ptr(address, i), _slang_vector_get_element(val, i)); \
        return result;                                                                            \
    }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, float, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, float, 4)
#endif
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, float, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, int, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, int, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, int, 4)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, uint, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, uint, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, uint, 4)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, ulonglong, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, ulonglong, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, ulonglong, 4)

template<typename T, int n>
struct GetVectorTypeImpl
{
};

#define GET_VECTOR_TYPE_IMPL(T, n)                                     \
    template<>                                                         \
    struct GetVectorTypeImpl<T, n>                                     \
    {                                                                  \
        typedef T##n type;                                             \
        static SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n fromScalar(T v) \
        {                                                              \
            return make_##T##n(v);                                     \
        }                                                              \
    };
#define GET_VECTOR_TYPE_IMPL_N(T) \
    GET_VECTOR_TYPE_IMPL(T, 1)    \
    GET_VECTOR_TYPE_IMPL(T, 2)    \
    GET_VECTOR_TYPE_IMPL(T, 3)    \
    GET_VECTOR_TYPE_IMPL(T, 4)

GET_VECTOR_TYPE_IMPL_N(int)
GET_VECTOR_TYPE_IMPL_N(uint)
GET_VECTOR_TYPE_IMPL_N(short)
GET_VECTOR_TYPE_IMPL_N(ushort)
GET_VECTOR_TYPE_IMPL_N(char)
GET_VECTOR_TYPE_IMPL_N(uchar)
GET_VECTOR_TYPE_IMPL_N(longlong)
GET_VECTOR_TYPE_IMPL_N(ulonglong)
GET_VECTOR_TYPE_IMPL_N(float)
GET_VECTOR_TYPE_IMPL_N(double)
#if SLANG_CUDA_ENABLE_HALF
GET_VECTOR_TYPE_IMPL_N(__half)
#endif
template<typename T, int n>
using Vector = typename GetVectorTypeImpl<T, n>::type;

template<typename T, int n, typename OtherT, int m>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Vector<T, n> _slang_vector_reshape(const Vector<OtherT, m> other)
{
    Vector<T, n> result;
    for (int i = 0; i < n; i++)
    {
        OtherT otherElement = T(0);
        if (i < m)
            otherElement = _slang_vector_get_element(other, i);
        *_slang_vector_get_element_ptr(&result, i) = (T)otherElement;
    }
    return result;
}

template<typename T, int ROWS, int COLS>
struct Matrix
{
    Vector<T, COLS> rows[ROWS];
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Vector<T, COLS>& operator[](size_t index)
    {
        return rows[index];
    }
};


template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(T scalar)
{
    Matrix<T, ROWS, COLS> result;
    for (int i = 0; i < ROWS; i++)
        result.rows[i] = GetVectorTypeImpl<T, COLS>::fromScalar(scalar);
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(const Vector<T, COLS>& row0)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Vector<T, COLS>& row0,
    const Vector<T, COLS>& row1)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    result.rows[1] = row1;
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Vector<T, COLS>& row0,
    const Vector<T, COLS>& row1,
    const Vector<T, COLS>& row2)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    result.rows[1] = row1;
    result.rows[2] = row2;
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Vector<T, COLS>& row0,
    const Vector<T, COLS>& row1,
    const Vector<T, COLS>& row2,
    const Vector<T, COLS>& row3)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    result.rows[1] = row1;
    result.rows[2] = row2;
    result.rows[3] = row3;
    return result;
}

template<typename T, int ROWS, int COLS, typename U, int otherRow, int otherCol>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Matrix<U, otherRow, otherCol>& other)
{
    Matrix<T, ROWS, COLS> result;
    int minRow = ROWS;
    int minCol = COLS;
    if (minRow > otherRow)
        minRow = otherRow;
    if (minCol > otherCol)
        minCol = otherCol;
    for (int i = 0; i < minRow; i++)
        for (int j = 0; j < minCol; j++)
            *_slang_vector_get_element_ptr(result.rows + i, j) =
                (T)_slang_vector_get_element(other.rows[i], j);
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(T v0, T v1, T v2, T v3)
{
    Matrix<T, ROWS, COLS> rs;
    rs.rows[0].x = v0;
    rs.rows[0].y = v1;
    rs.rows[1].x = v2;
    rs.rows[1].y = v3;
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5)
{
    Matrix<T, ROWS, COLS> rs;
    if (COLS == 3)
    {
        *_slang_vector_get_element_ptr(&rs.rows[0], 0) = v0;
        *_slang_vector_get_element_ptr(&rs.rows[0], 1) = v1;
        *_slang_vector_get_element_ptr(&rs.rows[0], 2) = v2;
        *_slang_vector_get_element_ptr(&rs.rows[1], 0) = v3;
        *_slang_vector_get_element_ptr(&rs.rows[1], 1) = v4;
        *_slang_vector_get_element_ptr(&rs.rows[1], 2) = v5;
    }
    else
    {
        rs.rows[0].x = v0;
        rs.rows[0].y = v1;
        rs.rows[1].x = v2;
        rs.rows[1].y = v3;
        rs.rows[2].x = v4;
        rs.rows[2].y = v5;
    }
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7)
{
    Matrix<T, ROWS, COLS> rs;
    if (COLS == 4)
    {
        *_slang_vector_get_element_ptr(&rs.rows[0], 0) = v0;
        *_slang_vector_get_element_ptr(&rs.rows[0], 1) = v1;
        *_slang_vector_get_element_ptr(&rs.rows[0], 2) = v2;
        *_slang_vector_get_element_ptr(&rs.rows[0], 3) = v3;
        *_slang_vector_get_element_ptr(&rs.rows[1], 0) = v4;
        *_slang_vector_get_element_ptr(&rs.rows[1], 1) = v5;
        *_slang_vector_get_element_ptr(&rs.rows[1], 2) = v6;
        *_slang_vector_get_element_ptr(&rs.rows[1], 3) = v7;
    }
    else
    {
        rs.rows[0].x = v0;
        rs.rows[0].y = v1;
        rs.rows[1].x = v2;
        rs.rows[1].y = v3;
        rs.rows[2].x = v4;
        rs.rows[2].y = v5;
        rs.rows[3].x = v6;
        rs.rows[3].y = v7;
    }
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7,
    T v8)
{
    Matrix<T, ROWS, COLS> rs;
    rs.rows[0].x = v0;
    rs.rows[0].y = v1;
    rs.rows[0].z = v2;
    rs.rows[1].x = v3;
    rs.rows[1].y = v4;
    rs.rows[1].z = v5;
    rs.rows[2].x = v6;
    rs.rows[2].y = v7;
    rs.rows[2].z = v8;
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7,
    T v8,
    T v9,
    T v10,
    T v11)
{
    Matrix<T, ROWS, COLS> rs;
    if (COLS == 4)
    {
        *_slang_vector_get_element_ptr(&rs.rows[0], 0) = v0;
        *_slang_vector_get_element_ptr(&rs.rows[0], 1) = v1;
        *_slang_vector_get_element_ptr(&rs.rows[0], 2) = v2;
        *_slang_vector_get_element_ptr(&rs.rows[0], 3) = v3;
        *_slang_vector_get_element_ptr(&rs.rows[1], 0) = v4;
        *_slang_vector_get_element_ptr(&rs.rows[1], 1) = v5;
        *_slang_vector_get_element_ptr(&rs.rows[1], 2) = v6;
        *_slang_vector_get_element_ptr(&rs.rows[1], 3) = v7;
        *_slang_vector_get_element_ptr(&rs.rows[2], 0) = v8;
        *_slang_vector_get_element_ptr(&rs.rows[2], 1) = v9;
        *_slang_vector_get_element_ptr(&rs.rows[2], 2) = v10;
        *_slang_vector_get_element_ptr(&rs.rows[2], 3) = v11;
    }
    else
    {
        rs.rows[0].x = v0;
        rs.rows[0].y = v1;
        rs.rows[0].z = v2;
        rs.rows[1].x = v3;
        rs.rows[1].y = v4;
        rs.rows[1].z = v5;
        rs.rows[2].x = v6;
        rs.rows[2].y = v7;
        rs.rows[2].z = v8;
        rs.rows[3].x = v9;
        rs.rows[3].y = v10;
        rs.rows[3].z = v11;
    }
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7,
    T v8,
    T v9,
    T v10,
    T v11,
    T v12,
    T v13,
    T v14,
    T v15)
{
    Matrix<T, ROWS, COLS> rs;
    rs.rows[0].x = v0;
    rs.rows[0].y = v1;
    rs.rows[0].z = v2;
    rs.rows[0].w = v3;
    rs.rows[1].x = v4;
    rs.rows[1].y = v5;
    rs.rows[1].z = v6;
    rs.rows[1].w = v7;
    rs.rows[2].x = v8;
    rs.rows[2].y = v9;
    rs.rows[2].z = v10;
    rs.rows[2].w = v11;
    rs.rows[3].x = v12;
    rs.rows[3].y = v13;
    rs.rows[3].z = v14;
    rs.rows[3].w = v15;
    return rs;
}

#define SLANG_MATRIX_BINARY_OP(T, op)                                   \
    template<int R, int C>                                              \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator op(     \
        const Matrix<T, R, C>& thisVal,                                 \
        const Matrix<T, R, C>& other)                                   \
    {                                                                   \
        Matrix<T, R, C> result;                                         \
        for (int i = 0; i < R; i++)                                     \
            for (int j = 0; j < C; j++)                                 \
                *_slang_vector_get_element_ptr(result.rows + i, j) =    \
                    _slang_vector_get_element(thisVal.rows[i], j)       \
                        op _slang_vector_get_element(other.rows[i], j); \
        return result;                                                  \
    }

#define SLANG_MATRIX_UNARY_OP(T, op)                                                               \
    template<int R, int C>                                                                         \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator op(const Matrix<T, R, C>& thisVal) \
    {                                                                                              \
        Matrix<T, R, C> result;                                                                    \
        for (int i = 0; i < R; i++)                                                                \
            for (int j = 0; j < C; j++)                                                            \
                *_slang_vector_get_element_ptr(result.rows + i, j) =                               \
                    op _slang_vector_get_element(thisVal.rows[i], j);                              \
        return result;                                                                             \
    }
#define SLANG_INT_MATRIX_OPS(T)   \
    SLANG_MATRIX_BINARY_OP(T, +)  \
    SLANG_MATRIX_BINARY_OP(T, -)  \
    SLANG_MATRIX_BINARY_OP(T, *)  \
    SLANG_MATRIX_BINARY_OP(T, /)  \
    SLANG_MATRIX_BINARY_OP(T, &)  \
    SLANG_MATRIX_BINARY_OP(T, |)  \
    SLANG_MATRIX_BINARY_OP(T, &&) \
    SLANG_MATRIX_BINARY_OP(T, ||) \
    SLANG_MATRIX_BINARY_OP(T, ^)  \
    SLANG_MATRIX_BINARY_OP(T, %)  \
    SLANG_MATRIX_UNARY_OP(T, !)   \
    SLANG_MATRIX_UNARY_OP(T, ~)
#define SLANG_FLOAT_MATRIX_OPS(T) \
    SLANG_MATRIX_BINARY_OP(T, +)  \
    SLANG_MATRIX_BINARY_OP(T, -)  \
    SLANG_MATRIX_BINARY_OP(T, *)  \
    SLANG_MATRIX_BINARY_OP(T, /)  \
    SLANG_MATRIX_UNARY_OP(T, -)
SLANG_INT_MATRIX_OPS(int)
SLANG_INT_MATRIX_OPS(uint)
SLANG_INT_MATRIX_OPS(short)
SLANG_INT_MATRIX_OPS(ushort)
SLANG_INT_MATRIX_OPS(char)
SLANG_INT_MATRIX_OPS(uchar)
SLANG_INT_MATRIX_OPS(longlong)
SLANG_INT_MATRIX_OPS(ulonglong)
SLANG_FLOAT_MATRIX_OPS(float)
SLANG_FLOAT_MATRIX_OPS(double)
#if SLANG_CUDA_ENABLE_HALF
SLANG_FLOAT_MATRIX_OPS(__half)
#endif
#define SLANG_MATRIX_INT_NEG_OP(T)                                                        \
    template<int R, int C>                                                                \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator-(Matrix<T, R, C> thisVal) \
    {                                                                                     \
        Matrix<T, R, C> result;                                                           \
        for (int i = 0; i < R; i++)                                                       \
            for (int j = 0; j < C; j++)                                                   \
                *_slang_vector_get_element_ptr(result.rows + i, j) =                      \
                    0 - _slang_vector_get_element(thisVal.rows[i], j);                    \
        return result;                                                                    \
    }
SLANG_MATRIX_INT_NEG_OP(int)
SLANG_MATRIX_INT_NEG_OP(uint)
SLANG_MATRIX_INT_NEG_OP(short)
SLANG_MATRIX_INT_NEG_OP(ushort)
SLANG_MATRIX_INT_NEG_OP(char)
SLANG_MATRIX_INT_NEG_OP(uchar)
SLANG_MATRIX_INT_NEG_OP(longlong)
SLANG_MATRIX_INT_NEG_OP(ulonglong)

#define SLANG_FLOAT_MATRIX_MOD(T)                                                 \
    template<int R, int C>                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator%(                 \
        Matrix<T, R, C> left,                                                     \
        Matrix<T, R, C> right)                                                    \
    {                                                                             \
        Matrix<T, R, C> result;                                                   \
        for (int i = 0; i < R; i++)                                               \
            for (int j = 0; j < C; j++)                                           \
                *_slang_vector_get_element_ptr(result.rows + i, j) = _slang_fmod( \
                    _slang_vector_get_element(left.rows[i], j),                   \
                    _slang_vector_get_element(right.rows[i], j));                 \
        return result;                                                            \
    }

SLANG_FLOAT_MATRIX_MOD(float)
SLANG_FLOAT_MATRIX_MOD(double)
#if SLANG_CUDA_ENABLE_HALF
template<int R, int C>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<__half, R, C> operator%(
    Matrix<__half, R, C> left,
    Matrix<__half, R, C> right)
{
    Matrix<__half, R, C> result;
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++)
            *_slang_vector_get_element_ptr(result.rows + i, j) = __float2half(_slang_fmod(
                __half2float(_slang_vector_get_element(left.rows[i], j)),
                __half2float(_slang_vector_get_element(right.rows[i], j))));
    return result;
}
#endif
#undef SLANG_FLOAT_MATRIX_MOD
#undef SLANG_MATRIX_BINARY_OP
#undef SLANG_MATRIX_UNARY_OP
#undef SLANG_INT_MATRIX_OPS
#undef SLANG_FLOAT_MATRIX_OPS
#undef SLANG_MATRIX_INT_NEG_OP
#undef SLANG_FLOAT_MATRIX_MOD

#define SLANG_SELECT_IMPL(T, N)                                                                  \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Vector<T, N> _slang_select(                               \
        bool##N condition,                                                                       \
        Vector<T, N> v0,                                                                         \
        Vector<T, N> v1)                                                                         \
    {                                                                                            \
        Vector<T, N> result;                                                                     \
        for (int i = 0; i < N; i++)                                                              \
        {                                                                                        \
            *_slang_vector_get_element_ptr(&result, i) = _slang_vector_get_element(condition, i) \
                                                             ? _slang_vector_get_element(v0, i)  \
                                                             : _slang_vector_get_element(v1, i); \
        }                                                                                        \
        return result;                                                                           \
    }
#define SLANG_SELECT_T(T)   \
    SLANG_SELECT_IMPL(T, 2) \
    SLANG_SELECT_IMPL(T, 3) \
    SLANG_SELECT_IMPL(T, 4)

SLANG_SELECT_T(int)
SLANG_SELECT_T(uint)
SLANG_SELECT_T(short)
SLANG_SELECT_T(ushort)
SLANG_SELECT_T(char)
SLANG_SELECT_T(uchar)
SLANG_SELECT_T(float)
SLANG_SELECT_T(double)

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_select(bool condition, T v0, T v1)
{
    return condition ? v0 : v1;
}

//
// Half support
//

#if SLANG_CUDA_ENABLE_HALF
SLANG_SELECT_T(__half)

// Convenience functions ushort -> half

SLANG_FORCE_INLINE SLANG_CUDA_CALL __half2 __ushort_as_half(const ushort2& i)
{
    return __halves2half2(__ushort_as_half(i.x), __ushort_as_half(i.y));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half3 __ushort_as_half(const ushort3& i)
{
    return __half3{__ushort_as_half(i.x), __ushort_as_half(i.y), __ushort_as_half(i.z)};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half4 __ushort_as_half(const ushort4& i)
{
    return __half4{
        __ushort_as_half(i.x),
        __ushort_as_half(i.y),
        __ushort_as_half(i.z),
        __ushort_as_half(i.w)};
}

// Convenience functions half -> ushort

SLANG_FORCE_INLINE SLANG_CUDA_CALL ushort2 __half_as_ushort(const __half2& i)
{
    return make_ushort2(__half_as_ushort(i.x), __half_as_ushort(i.y));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL ushort3 __half_as_ushort(const __half3& i)
{
    return make_ushort3(__half_as_ushort(i.x), __half_as_ushort(i.y), __half_as_ushort(i.z));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL ushort4 __half_as_ushort(const __half4& i)
{
    return make_ushort4(
        __half_as_ushort(i.x),
        __half_as_ushort(i.y),
        __half_as_ushort(i.z),
        __half_as_ushort(i.w));
}

// This is a little bit of a hack. Fortunately CUDA has the definitions of the templated types in
// include/surface_indirect_functions.h
// Here we find the template definition requires a specialization of __nv_isurf_trait to allow
// a specialization of the surface write functions.
// This *isn't* a problem on the read functions as they don't have a return type that uses this
// mechanism

template<>
struct __nv_isurf_trait<__half>
{
    typedef void type;
};
template<>
struct __nv_isurf_trait<__half2>
{
    typedef void type;
};
template<>
struct __nv_isurf_trait<__half4>
{
    typedef void type;
};

#define SLANG_DROP_PARENS(...) __VA_ARGS__

#define SLANG_SURFACE_READ(FUNC_NAME, TYPE_ARGS, ARGS)                                             \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL __half FUNC_NAME<__half>(                                   \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        return __ushort_as_half(FUNC_NAME<ushort>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL __half2 FUNC_NAME<__half2>(                                 \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        return __ushort_as_half(                                                                   \
            FUNC_NAME<ushort2>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode));                    \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL __half4 FUNC_NAME<__half4>(                                 \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        return __ushort_as_half(                                                                   \
            FUNC_NAME<ushort4>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode));                    \
    }

SLANG_SURFACE_READ(surf1Dread, (int x), (x))
SLANG_SURFACE_READ(surf2Dread, (int x, int y), (x, y))
SLANG_SURFACE_READ(surf3Dread, (int x, int y, int z), (x, y, z))
SLANG_SURFACE_READ(surf1DLayeredread, (int x, int layer), (x, layer))
SLANG_SURFACE_READ(surf2DLayeredread, (int x, int y, int layer), (x, y, layer))
SLANG_SURFACE_READ(surfCubemapread, (int x, int y, int face), (x, y, face))
SLANG_SURFACE_READ(surfCubemapLayeredread, (int x, int y, int layerFace), (x, y, layerFace))

#define SLANG_SURFACE_WRITE(FUNC_NAME, TYPE_ARGS, ARGS)                                            \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void FUNC_NAME<__half>(                                     \
        __half data,                                                                               \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        FUNC_NAME<ushort>(__half_as_ushort(data), surfObj, SLANG_DROP_PARENS ARGS, boundaryMode);  \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void FUNC_NAME<__half2>(                                    \
        __half2 data,                                                                              \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        FUNC_NAME<ushort2>(__half_as_ushort(data), surfObj, SLANG_DROP_PARENS ARGS, boundaryMode); \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void FUNC_NAME<__half4>(                                    \
        __half4 data,                                                                              \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        FUNC_NAME<ushort4>(__half_as_ushort(data), surfObj, SLANG_DROP_PARENS ARGS, boundaryMode); \
    }

SLANG_SURFACE_WRITE(surf1Dwrite, (int x), (x))
SLANG_SURFACE_WRITE(surf2Dwrite, (int x, int y), (x, y))
SLANG_SURFACE_WRITE(surf3Dwrite, (int x, int y, int z), (x, y, z))
SLANG_SURFACE_WRITE(surf1DLayeredwrite, (int x, int layer), (x, layer))
SLANG_SURFACE_WRITE(surf2DLayeredwrite, (int x, int y, int layer), (x, y, layer))
SLANG_SURFACE_WRITE(surfCubemapwrite, (int x, int y, int face), (x, y, face))
SLANG_SURFACE_WRITE(surfCubemapLayeredwrite, (int x, int y, int layerFace), (x, y, layerFace))

// ! Hack to test out reading !!!
// Only works converting *from* half

// template <typename T>
// SLANG_FORCE_INLINE SLANG_CUDA_CALL T surf2Dread_convert(cudaSurfaceObject_t surfObj, int x, int
// y, cudaSurfaceBoundaryMode boundaryMode);

#define SLANG_SURFACE_READ_HALF_CONVERT(FUNC_NAME, TYPE_ARGS, ARGS)                              \
                                                                                                 \
    template<typename T>                                                                         \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T FUNC_NAME##_convert(                                    \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode);                                                   \
                                                                                                 \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL float FUNC_NAME##_convert<float>(                         \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        return __ushort_as_half(                                                                 \
            FUNC_NAME<uint16_t>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode));                 \
    }                                                                                            \
                                                                                                 \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL float2 FUNC_NAME##_convert<float2>(                       \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        const __half2 v =                                                                        \
            __ushort_as_half(FUNC_NAME<ushort2>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
        return float2{v.x, v.y};                                                                 \
    }                                                                                            \
                                                                                                 \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL float4 FUNC_NAME##_convert<float4>(                       \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        const __half4 v =                                                                        \
            __ushort_as_half(FUNC_NAME<ushort4>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
        return float4{v.x, v.y, v.z, v.w};                                                       \
    }

SLANG_SURFACE_READ_HALF_CONVERT(surf1Dread, (int x), (x))
SLANG_SURFACE_READ_HALF_CONVERT(surf2Dread, (int x, int y), (x, y))
SLANG_SURFACE_READ_HALF_CONVERT(surf3Dread, (int x, int y, int z), (x, y, z))

#endif

// Support for doing format conversion when writing to a surface/RWTexture

// NOTE! For normal surface access x values are *byte* addressed.
// For the _convert versions they are *not*. They don't need to be because sust.p does not require
// it.

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert(
    T,
    cudaSurfaceObject_t surfObj,
    int x,
    cudaSurfaceBoundaryMode boundaryMode);
template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert(
    T,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    cudaSurfaceBoundaryMode boundaryMode);
template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert(
    T,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    int z,
    cudaSurfaceBoundaryMode boundaryMode);

// https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-sust

// Float

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert<float>(
    float v,
    cudaSurfaceObject_t surfObj,
    int x,
    cudaSurfaceBoundaryMode boundaryMode)
{
    asm volatile(
        "{sust.p.1d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1}], {%2};}\n\t" ::"l"(surfObj),
        "r"(x),
        "f"(v));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert<float>(
    float v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    cudaSurfaceBoundaryMode boundaryMode)
{
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1,%2}], {%3};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "f"(v));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert<float>(
    float v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    int z,
    cudaSurfaceBoundaryMode boundaryMode)
{
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1,%2,%3}], {%4};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "r"(z),
        "f"(v));
}

// Float2

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert<float2>(
    float2 v,
    cudaSurfaceObject_t surfObj,
    int x,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y;
    asm volatile(
        "{sust.p.1d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1}], {%2,%3};}\n\t" ::"l"(surfObj),
        "r"(x),
        "f"(vx),
        "f"(vy));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert<float2>(
    float2 v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y;
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1,%2}], {%3,%4};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "f"(vx),
        "f"(vy));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert<float2>(
    float2 v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    int z,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y;
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1,%2,%3}], {%4,%5};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "r"(z),
        "f"(vx),
        "f"(vy));
}

// Float4
template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert<float4>(
    float4 v,
    cudaSurfaceObject_t surfObj,
    int x,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y, vz = v.z, vw = v.w;
    asm volatile(
        "{sust.p.1d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1}], {%2,%3,%4,%5};}\n\t" ::"l"(surfObj),
        "r"(x),
        "f"(vx),
        "f"(vy),
        "f"(vz),
        "f"(vw));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert<float4>(
    float4 v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y, vz = v.z, vw = v.w;
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE
        " [%0, {%1,%2}], {%3,%4,%5,%6};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "f"(vx),
        "f"(vy),
        "f"(vz),
        "f"(vw));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert<float4>(
    float4 v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    int z,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y, vz = v.z, vw = v.w;
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE
        " [%0, {%1,%2,%3}], {%4,%5,%6,%7};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "r"(z),
        "f"(vx),
        "f"(vy),
        "f"(vz),
        "f"(vw));
}

// ----------------------------- F32 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_ceil(float f)
{
    return ::ceilf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_floor(float f)
{
    return ::floorf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_round(float f)
{
    return ::roundf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sin(float f)
{
    return ::sinf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_cos(float f)
{
    return ::cosf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL void F32_sincos(float f, float* s, float* c)
{
    ::sincosf(f, s, c);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_tan(float f)
{
    return ::tanf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_asin(float f)
{
    return ::asinf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_acos(float f)
{
    return ::acosf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_atan(float f)
{
    return ::atanf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sinh(float f)
{
    return ::sinhf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_cosh(float f)
{
    return ::coshf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_tanh(float f)
{
    return ::tanhf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_log2(float f)
{
    return ::log2f(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_log(float f)
{
    return ::logf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_log10(float f)
{
    return ::log10f(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_exp2(float f)
{
    return ::exp2f(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_exp(float f)
{
    return ::expf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_abs(float f)
{
    return ::fabsf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_trunc(float f)
{
    return ::truncf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sqrt(float f)
{
    return ::sqrtf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_rsqrt(float f)
{
    return ::rsqrtf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sign(float f)
{
    return (f == 0.0f) ? f : ((f < 0.0f) ? -1.0f : 1.0f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_frac(float f)
{
    return f - F32_floor(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F32_isnan(float f)
{
    return isnan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F32_isfinite(float f)
{
    return isfinite(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F32_isinf(float f)
{
    return isinf(f);
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_min(float a, float b)
{
    return ::fminf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_max(float a, float b)
{
    return ::fmaxf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_pow(float a, float b)
{
    return ::powf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_fmod(float a, float b)
{
    return ::fmodf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_remainder(float a, float b)
{
    return ::remainderf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_atan2(float a, float b)
{
    return float(::atan2(a, b));
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_frexp(float x, int* e)
{
    return frexpf(x, e);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_modf(float x, float* ip)
{
    return ::modff(x, ip);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t F32_asuint(float f)
{
    Union32 u;
    u.f = f;
    return u.u;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t F32_asint(float f)
{
    Union32 u;
    u.f = f;
    return u.i;
}

// Ternary
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_fma(float a, float b, float c)
{
    return ::fmaf(a, b, c);
}


// ----------------------------- F64 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_ceil(double f)
{
    return ::ceil(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_floor(double f)
{
    return ::floor(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_round(double f)
{
    return ::round(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sin(double f)
{
    return ::sin(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_cos(double f)
{
    return ::cos(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL void F64_sincos(double f, double* s, double* c)
{
    ::sincos(f, s, c);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_tan(double f)
{
    return ::tan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_asin(double f)
{
    return ::asin(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_acos(double f)
{
    return ::acos(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_atan(double f)
{
    return ::atan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sinh(double f)
{
    return ::sinh(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_cosh(double f)
{
    return ::cosh(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_tanh(double f)
{
    return ::tanh(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_log2(double f)
{
    return ::log2(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_log(double f)
{
    return ::log(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_log10(float f)
{
    return ::log10(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_exp2(double f)
{
    return ::exp2(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_exp(double f)
{
    return ::exp(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_abs(double f)
{
    return ::fabs(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_trunc(double f)
{
    return ::trunc(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sqrt(double f)
{
    return ::sqrt(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_rsqrt(double f)
{
    return ::rsqrt(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sign(double f)
{
    return (f == 0.0) ? f : ((f < 0.0) ? -1.0 : 1.0);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_frac(double f)
{
    return f - F64_floor(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F64_isnan(double f)
{
    return isnan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F64_isfinite(double f)
{
    return isfinite(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F64_isinf(double f)
{
    return isinf(f);
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_min(double a, double b)
{
    return ::fmin(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_max(double a, double b)
{
    return ::fmax(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_pow(double a, double b)
{
    return ::pow(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_fmod(double a, double b)
{
    return ::fmod(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_remainder(double a, double b)
{
    return ::remainder(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_atan2(double a, double b)
{
    return ::atan2(a, b);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_frexp(double x, int* e)
{
    return ::frexp(x, e);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_modf(double x, double* ip)
{
    return ::modf(x, ip);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL void F64_asuint(double d, uint32_t* low, uint32_t* hi)
{
    Union64 u;
    u.d = d;
    *low = uint32_t(u.u);
    *hi = uint32_t(u.u >> 32);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL void F64_asint(double d, int32_t* low, int32_t* hi)
{
    Union64 u;
    u.d = d;
    *low = int32_t(u.u);
    *hi = int32_t(u.u >> 32);
}

// Ternary
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_fma(double a, double b, double c)
{
    return ::fma(a, b, c);
}

// ----------------------------- I32 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t I32_abs(int32_t f)
{
    return (f < 0) ? -f : f;
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t I32_min(int32_t a, int32_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t I32_max(int32_t a, int32_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float I32_asfloat(int32_t x)
{
    Union32 u;
    u.i = x;
    return u.f;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t I32_asuint(int32_t x)
{
    return uint32_t(x);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double I32_asdouble(int32_t low, int32_t hi)
{
    Union64 u;
    u.u = (uint64_t(hi) << 32) | uint32_t(low);
    return u.d;
}

// ----------------------------- U32 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_abs(uint32_t f)
{
    return f;
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_min(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_max(uint32_t a, uint32_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float U32_asfloat(uint32_t x)
{
    Union32 u;
    u.u = x;
    return u.f;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_asint(int32_t x)
{
    return uint32_t(x);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL double U32_asdouble(uint32_t low, uint32_t hi)
{
    Union64 u;
    u.u = (uint64_t(hi) << 32) | low;
    return u.d;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_countbits(uint32_t v)
{
    // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html#group__CUDA__MATH__INTRINSIC__INT_1g43c9c7d2b9ebf202ff1ef5769989be46
    return __popc(v);
}


// ----------------------------- I64 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t I64_abs(int64_t f)
{
    return (f < 0) ? -f : f;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t I64_min(int64_t a, int64_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t I64_max(int64_t a, int64_t b)
{
    return a > b ? a : b;
}

// ----------------------------- U64 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t U64_abs(uint64_t f)
{
    return f;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t U64_min(uint64_t a, uint64_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t U64_max(uint64_t a, uint64_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U64_countbits(uint64_t v)
{
    // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html#group__CUDA__MATH__INTRINSIC__INT_1g43c9c7d2b9ebf202ff1ef5769989be46
    return __popcll(v);
}


// ----------------------------- ResourceType -----------------------------------------


// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-structuredbuffer-getdimensions
// Missing  Load(_In_  int  Location, _Out_ uint Status);

template<typename T>
struct StructuredBuffer
{
    SLANG_CUDA_CALL const T& operator[](size_t index) const
    {
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
        SLANG_BOUND_CHECK(index, count);
#endif
        return data[index];
    }

    SLANG_CUDA_CALL const T& Load(size_t index) const
    {
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
        SLANG_BOUND_CHECK(index, count);
#endif
        return data[index];
    }

#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
    SLANG_CUDA_CALL void GetDimensions(uint32_t* outNumStructs, uint32_t* outStride)
    {
        *outNumStructs = uint32_t(count);
        *outStride = uint32_t(sizeof(T));
    }
#endif

    T* data;
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
    size_t count;
#endif
};

template<typename T>
struct RWStructuredBuffer : StructuredBuffer<T>
{
    SLANG_CUDA_CALL T& operator[](size_t index) const
    {
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
        SLANG_BOUND_CHECK(index, this->count);
#endif
        return this->data[index];
    }
};

// Missing  Load(_In_  int  Location, _Out_ uint Status);
struct ByteAddressBuffer
{
    SLANG_CUDA_CALL void GetDimensions(uint32_t* outDim) const { *outDim = uint32_t(sizeInBytes); }
    SLANG_CUDA_CALL uint32_t Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        return data[index >> 2];
    }
    SLANG_CUDA_CALL uint2 Load2(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint2{data[dataIdx], data[dataIdx + 1]};
    }
    SLANG_CUDA_CALL uint3 Load3(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint3{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2]};
    }
    SLANG_CUDA_CALL uint4 Load4(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint4{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2], data[dataIdx + 3]};
    }
    template<typename T>
    SLANG_CUDA_CALL T Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        T data;
        memcpy(&data, ((const char*)this->data) + index, sizeof(T));
        return data;
    }
    template<typename T>
    SLANG_CUDA_CALL StructuredBuffer<T> asStructuredBuffer() const
    {
        StructuredBuffer<T> rs;
        rs.data = (T*)data;
        rs.count = sizeInBytes / sizeof(T);
        return rs;
    }
    const uint32_t* data;
    size_t sizeInBytes; //< Must be multiple of 4
};

// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-rwbyteaddressbuffer
// Missing support for Atomic operations
// Missing support for Load with status
struct RWByteAddressBuffer
{
    SLANG_CUDA_CALL void GetDimensions(uint32_t* outDim) const { *outDim = uint32_t(sizeInBytes); }

    SLANG_CUDA_CALL uint32_t Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        return data[index >> 2];
    }
    SLANG_CUDA_CALL uint2 Load2(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint2{data[dataIdx], data[dataIdx + 1]};
    }
    SLANG_CUDA_CALL uint3 Load3(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint3{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2]};
    }
    SLANG_CUDA_CALL uint4 Load4(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint4{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2], data[dataIdx + 3]};
    }
    template<typename T>
    SLANG_CUDA_CALL T Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        T data;
        memcpy(&data, ((const char*)this->data) + index, sizeof(T));
        return data;
    }

    SLANG_CUDA_CALL void Store(size_t index, uint32_t v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        data[index >> 2] = v;
    }
    SLANG_CUDA_CALL void Store2(size_t index, uint2 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
    }
    SLANG_CUDA_CALL void Store3(size_t index, uint3 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
        data[dataIdx + 2] = v.z;
    }
    SLANG_CUDA_CALL void Store4(size_t index, uint4 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
        data[dataIdx + 2] = v.z;
        data[dataIdx + 3] = v.w;
    }
    template<typename T>
    SLANG_CUDA_CALL void Store(size_t index, T const& value) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        memcpy((char*)data + index, &value, sizeof(T));
    }

    /// Can be used in the core module to gain access
    template<typename T>
    SLANG_CUDA_CALL T* _getPtrAt(size_t index)
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        return (T*)(((char*)data) + index);
    }
    template<typename T>
    SLANG_CUDA_CALL RWStructuredBuffer<T> asStructuredBuffer() const
    {
        RWStructuredBuffer<T> rs;
        rs.data = (T*)data;
        rs.count = sizeInBytes / sizeof(T);
        return rs;
    }
    uint32_t* data;
    size_t sizeInBytes; //< Must be multiple of 4
};


// ---------------------- Wave --------------------------------------

// TODO(JS): It appears that cuda does not have a simple way to get a lane index.
//
// Another approach could be...
// laneId = ((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x) &
// SLANG_CUDA_WARP_MASK If that is really true another way to do this, would be for code generator
// to add this function with the [numthreads] baked in.
//
// For now I'll just assume you have a launch that makes the following correct if the kernel uses
// WaveGetLaneIndex()
#ifndef SLANG_USE_ASM_LANE_ID
__forceinline__ __device__ uint32_t _getLaneId()
{
    // If the launch is (or I guess some multiple of the warp size)
    // we try this mechanism, which is apparently faster.
    return threadIdx.x & SLANG_CUDA_WARP_MASK;
}
#else
__forceinline__ __device__ uint32_t _getLaneId()
{
    // https://stackoverflow.com/questions/44337309/whats-the-most-efficient-way-to-calculate-the-warp-id-lane-id-in-a-1-d-grid#
    // This mechanism is not the fastest way to do it, and that is why the other mechanism
    // is the default. But the other mechanism relies on a launch that makes the assumption
    // true.
    unsigned ret;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}
#endif

typedef int WarpMask;

// It appears that the __activemask() cannot always be used because
// threads need to be converged.
//
// For CUDA the article claims mask has to be used carefully
// https://devblogs.nvidia.com/using-cuda-warp-level-primitives/
// With the Warp intrinsics there is no mask, and it's just the 'active lanes'.
// __activemask() though does not require there is convergence, so that doesn't work.
//
// '__ballot_sync' produces a convergance.
//
// From the CUDA docs:
// ```For __all_sync, __any_sync, and __ballot_sync, a mask must be passed that specifies the
// threads participating in the call. A bit, representing the thread's lane ID, must be set for each
// participating thread to ensure they are properly converged before the intrinsic is executed by
// the hardware. All active threads named in mask must execute the same intrinsic with the same
// mask, or the result is undefined.```
//
// Currently there isn't a mechanism to correctly get the mask without it being passed through.
// Doing so will most likely require some changes to slang code generation to track masks, for now
// then we use _getActiveMask.

// Return mask of all the lanes less than the current lane
__forceinline__ __device__ WarpMask _getLaneLtMask()
{
    return (int(1) << _getLaneId()) - 1;
}

// TODO(JS):
// THIS IS NOT CORRECT! That determining the appropriate active mask requires appropriate
// mask tracking.
__forceinline__ __device__ WarpMask _getActiveMask()
{
    return __ballot_sync(__activemask(), true);
}

// Return a mask suitable for the 'MultiPrefix' style functions
__forceinline__ __device__ WarpMask _getMultiPrefixMask(int mask)
{
    return mask;
}

// Note! Note will return true if mask is 0, but thats okay, because there must be one
// lane active to execute anything
__inline__ __device__ bool _waveIsSingleLane(WarpMask mask)
{
    return (mask & (mask - 1)) == 0;
}

// Returns the power of 2 size of run of set bits. Returns 0 if not a suitable run.
// Examples:
// 0b00000000'00000000'00000000'11111111 -> 8
// 0b11111111'11111111'11111111'11111111 -> 32
// 0b00000000'00000000'00000000'00011111 -> 0 (since 5 is not a power of 2)
// 0b00000000'00000000'00000000'11110000 -> 0 (since the run of bits does not start at the LSB)
// 0b00000000'00000000'00000000'00100111 -> 0 (since it is not a single contiguous run)
__inline__ __device__ int _waveCalcPow2Offset(WarpMask mask)
{
    // This should be the most common case, so fast path it
    if (mask == SLANG_CUDA_WARP_BITMASK)
    {
        return SLANG_CUDA_WARP_SIZE;
    }
    // Is it a contiguous run of bits?
    if ((mask & (mask + 1)) == 0)
    {
        // const int offsetSize = __ffs(mask + 1) - 1;
        const int offset = 32 - __clz(mask);
        // Is it a power of 2 size
        if ((offset & (offset - 1)) == 0)
        {
            return offset;
        }
    }
    return 0;
}

__inline__ __device__ bool _waveIsFirstLane()
{
    const WarpMask mask = __activemask();
    // We special case bit 0, as that most warps are expected to be fully active.

    // mask & -mask, isolates the lowest set bit.
    // return (mask & 1 ) || ((mask & -mask) == (1 << _getLaneId()));

    // This mechanism is most similar to what was in an nVidia post, so assume it is prefered.
    return (mask & 1) || ((__ffs(mask) - 1) == _getLaneId());
}

template<typename T>
struct WaveOpOr
{
    __inline__ __device__ static T getInitial(T a) { return 0; }
    __inline__ __device__ static T doOp(T a, T b) { return a | b; }
};

template<typename T>
struct WaveOpAnd
{
    __inline__ __device__ static T getInitial(T a) { return ~T(0); }
    __inline__ __device__ static T doOp(T a, T b) { return a & b; }
};

template<typename T>
struct WaveOpXor
{
    __inline__ __device__ static T getInitial(T a) { return 0; }
    __inline__ __device__ static T doOp(T a, T b) { return a ^ b; }
    __inline__ __device__ static T doInverse(T a, T b) { return a ^ b; }
};

template<typename T>
struct WaveOpAdd
{
    __inline__ __device__ static T getInitial(T a) { return 0; }
    __inline__ __device__ static T doOp(T a, T b) { return a + b; }
    __inline__ __device__ static T doInverse(T a, T b) { return a - b; }
};

template<typename T>
struct WaveOpMul
{
    __inline__ __device__ static T getInitial(T a) { return T(1); }
    __inline__ __device__ static T doOp(T a, T b) { return a * b; }
    // Using this inverse for int is probably undesirable - because in general it requires T to have
    // more precision There is also a performance aspect to it, where divides are generally
    // significantly slower
    __inline__ __device__ static T doInverse(T a, T b) { return a / b; }
};

template<typename T>
struct WaveOpMax
{
    __inline__ __device__ static T getInitial(T a) { return a; }
    __inline__ __device__ static T doOp(T a, T b) { return a > b ? a : b; }
};

template<typename T>
struct WaveOpMin
{
    __inline__ __device__ static T getInitial(T a) { return a; }
    __inline__ __device__ static T doOp(T a, T b) { return a < b ? a : b; }
};

template<typename T>
struct ElementTypeTrait;

// Scalar
template<>
struct ElementTypeTrait<int>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<uint>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<float>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<double>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<uint64_t>
{
    typedef uint64_t Type;
};
template<>
struct ElementTypeTrait<int64_t>
{
    typedef int64_t Type;
};

// Vector
template<>
struct ElementTypeTrait<int1>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<int2>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<int3>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<int4>
{
    typedef int Type;
};

template<>
struct ElementTypeTrait<uint1>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<uint2>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<uint3>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<uint4>
{
    typedef uint Type;
};

template<>
struct ElementTypeTrait<float1>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<float2>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<float3>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<float4>
{
    typedef float Type;
};

template<>
struct ElementTypeTrait<double1>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<double2>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<double3>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<double4>
{
    typedef double Type;
};

// Matrix
template<typename T, int ROWS, int COLS>
struct ElementTypeTrait<Matrix<T, ROWS, COLS>>
{
    typedef T Type;
};

// Scalar
template<typename INTF, typename T>
__device__ T _waveReduceScalar(WarpMask mask, T val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);
    if (offsetSize > 0)
    {
        // Fast path O(log2(activeLanes))
        for (int offset = offsetSize >> 1; offset > 0; offset >>= 1)
        {
            val = INTF::doOp(val, __shfl_xor_sync(mask, val, offset));
        }
    }
    else if (!_waveIsSingleLane(mask))
    {
        T result = INTF::getInitial(val);
        int remaining = mask;
        while (remaining)
        {
            const int laneBit = remaining & -remaining;
            // Get the sourceLane
            const int srcLane = __ffs(laneBit) - 1;
            // Broadcast (can also broadcast to self)
            result = INTF::doOp(result, __shfl_sync(mask, val, srcLane));
            remaining &= ~laneBit;
        }
        return result;
    }
    return val;
}


// Multiple values
template<typename INTF, typename T, size_t COUNT>
__device__ void _waveReduceMultiple(WarpMask mask, T* val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);
    if (offsetSize > 0)
    {
        // Fast path O(log2(activeLanes))
        for (int offset = offsetSize >> 1; offset > 0; offset >>= 1)
        {
            for (size_t i = 0; i < COUNT; ++i)
            {
                val[i] = INTF::doOp(val[i], __shfl_xor_sync(mask, val[i], offset));
            }
        }
    }
    else if (!_waveIsSingleLane(mask))
    {
        // Copy the original
        T originalVal[COUNT];
        for (size_t i = 0; i < COUNT; ++i)
        {
            const T v = val[i];
            originalVal[i] = v;
            val[i] = INTF::getInitial(v);
        }

        int remaining = mask;
        while (remaining)
        {
            const int laneBit = remaining & -remaining;
            // Get the sourceLane
            const int srcLane = __ffs(laneBit) - 1;
            // Broadcast (can also broadcast to self)
            for (size_t i = 0; i < COUNT; ++i)
            {
                val[i] = INTF::doOp(val[i], __shfl_sync(mask, originalVal[i], srcLane));
            }
            remaining &= ~laneBit;
        }
    }
}

template<typename INTF, typename T>
__device__ void _waveReduceMultiple(WarpMask mask, T* val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<INTF, ElemType, sizeof(T) / sizeof(ElemType)>(mask, (ElemType*)val);
}

template<typename T>
__inline__ __device__ T _waveOr(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpOr<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveAnd(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpAnd<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveXor(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpXor<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveProduct(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpMul<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveSum(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpAdd<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveMin(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpMin<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveMax(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpMax<T>, T>(mask, val);
}

// Fast-path specializations when CUDA warp reduce operators are available
#if __CUDA_ARCH__ >= 800 // 8.x or higher
template<>
__inline__ __device__ unsigned _waveOr<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_or_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveAnd<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_and_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveXor<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_xor_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveSum<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_add_sync(mask, val);
}

template<>
__inline__ __device__ int _waveSum<int>(WarpMask mask, int val)
{
    return __reduce_add_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveMin<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_min_sync(mask, val);
}

template<>
__inline__ __device__ int _waveMin<int>(WarpMask mask, int val)
{
    return __reduce_min_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveMax<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_max_sync(mask, val);
}

template<>
__inline__ __device__ int _waveMax<int>(WarpMask mask, int val)
{
    return __reduce_max_sync(mask, val);
}
#endif


// Multiple

template<typename T>
__inline__ __device__ T _waveOrMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpOr<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveAndMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpAnd<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveXorMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpXor<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveProductMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpMul<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveSumMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpAdd<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveMinMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpMin<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveMaxMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpMax<ElemType>>(mask, &val);
    return val;
}


template<typename T>
__inline__ __device__ bool _waveAllEqual(WarpMask mask, T val)
{
    int pred;
    __match_all_sync(mask, val, &pred);
    return pred != 0;
}

template<typename T>
__inline__ __device__ bool _waveAllEqualMultiple(WarpMask mask, T inVal)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    int pred;
    const ElemType* src = (const ElemType*)&inVal;
    for (size_t i = 0; i < count; ++i)
    {
        __match_all_sync(mask, src[i], &pred);
        if (pred == 0)
        {
            return false;
        }
    }
    return true;
}

template<typename T>
__inline__ __device__ T _waveReadFirst(WarpMask mask, T val)
{
    const int lowestLaneId = __ffs(mask) - 1;
    return __shfl_sync(mask, val, lowestLaneId);
}

template<typename T>
__inline__ __device__ T _waveReadFirstMultiple(WarpMask mask, T inVal)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    T outVal;
    const ElemType* src = (const ElemType*)&inVal;
    ElemType* dst = (ElemType*)&outVal;
    const int lowestLaneId = __ffs(mask) - 1;
    for (size_t i = 0; i < count; ++i)
    {
        dst[i] = __shfl_sync(mask, src[i], lowestLaneId);
    }
    return outVal;
}

template<typename T>
__inline__ __device__ T _waveShuffleMultiple(WarpMask mask, T inVal, int lane)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    T outVal;
    const ElemType* src = (const ElemType*)&inVal;
    ElemType* dst = (ElemType*)&outVal;
    for (size_t i = 0; i < count; ++i)
    {
        dst[i] = __shfl_sync(mask, src[i], lane);
    }
    return outVal;
}

// Scalar

// Invertable means that when we get to the end of the reduce, we can remove val (to make
// exclusive), using the inverse of the op.
template<typename INTF, typename T>
__device__ T _wavePrefixInvertableScalar(WarpMask mask, T val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();
    T result;
    if (offsetSize > 0)
    {
        // Sum is calculated inclusive of this lanes value
        result = val;
        for (int i = 1; i < offsetSize; i += i)
        {
            const T readVal = __shfl_up_sync(mask, result, i, offsetSize);
            if (laneId >= i)
            {
                result = INTF::doOp(result, readVal);
            }
        }
        // Remove val from the result, by applyin inverse
        result = INTF::doInverse(result, val);
    }
    else
    {
        result = INTF::getInitial(val);
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;
                // Broadcast (can also broadcast to self)
                const T readValue = __shfl_sync(mask, val, srcLane);
                // Only accumulate if srcLane is less than this lane
                if (srcLane < laneId)
                {
                    result = INTF::doOp(result, readValue);
                }
                remaining &= ~laneBit;
            }
        }
    }
    return result;
}


// This implementation separately tracks the value to be propogated, and the value
// that is the final result
template<typename INTF, typename T>
__device__ T _wavePrefixScalar(WarpMask mask, T val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();
    T result = INTF::getInitial(val);
    if (offsetSize > 0)
    {
        // For transmitted value we will do it inclusively with this lanes value
        // For the result we do not include the lanes value. This means an extra multiply for each
        // iteration but means we don't need to have a divide at the end and also removes overflow
        // issues in that scenario.
        for (int i = 1; i < offsetSize; i += i)
        {
            const T readVal = __shfl_up_sync(mask, val, i, offsetSize);
            if (laneId >= i)
            {
                result = INTF::doOp(result, readVal);
                val = INTF::doOp(val, readVal);
            }
        }
    }
    else
    {
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;
                // Broadcast (can also broadcast to self)
                const T readValue = __shfl_sync(mask, val, srcLane);
                // Only accumulate if srcLane is less than this lane
                if (srcLane < laneId)
                {
                    result = INTF::doOp(result, readValue);
                }
                remaining &= ~laneBit;
            }
        }
    }
    return result;
}


template<typename INTF, typename T, size_t COUNT>
__device__ T _waveOpCopy(T* dst, const T* src)
{
    for (size_t j = 0; j < COUNT; ++j)
    {
        dst[j] = src[j];
    }
}


template<typename INTF, typename T, size_t COUNT>
__device__ T _waveOpDoInverse(T* inOut, const T* val)
{
    for (size_t j = 0; j < COUNT; ++j)
    {
        inOut[j] = INTF::doInverse(inOut[j], val[j]);
    }
}

template<typename INTF, typename T, size_t COUNT>
__device__ T _waveOpSetInitial(T* out, const T* val)
{
    for (size_t j = 0; j < COUNT; ++j)
    {
        out[j] = INTF::getInitial(val[j]);
    }
}

template<typename INTF, typename T, size_t COUNT>
__device__ T _wavePrefixInvertableMultiple(WarpMask mask, T* val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();
    T originalVal[COUNT];
    _waveOpCopy<INTF, T, COUNT>(originalVal, val);

    if (offsetSize > 0)
    {
        // Sum is calculated inclusive of this lanes value
        for (int i = 1; i < offsetSize; i += i)
        {
            // TODO(JS): Note that here I don't split the laneId outside so it's only tested once.
            // This may be better but it would also mean that there would be shfl between lanes
            // that are on different (albeit identical) instructions. So this seems more likely to
            // work as expected with everything in lock step.
            for (size_t j = 0; j < COUNT; ++j)
            {
                const T readVal = __shfl_up_sync(mask, val[j], i, offsetSize);
                if (laneId >= i)
                {
                    val[j] = INTF::doOp(val[j], readVal);
                }
            }
        }
        // Remove originalVal from the result, by applyin inverse
        _waveOpDoInverse<INTF, T, COUNT>(val, originalVal);
    }
    else
    {
        _waveOpSetInitial<INTF, T, COUNT>(val, val);
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;

                for (size_t j = 0; j < COUNT; ++j)
                {
                    // Broadcast (can also broadcast to self)
                    const T readValue = __shfl_sync(mask, originalVal[j], srcLane);
                    // Only accumulate if srcLane is less than this lane
                    if (srcLane < laneId)
                    {
                        val[j] = INTF::doOp(val[j], readValue);
                    }
                    remaining &= ~laneBit;
                }
            }
        }
    }
}

template<typename INTF, typename T, size_t COUNT>
__device__ T _wavePrefixMultiple(WarpMask mask, T* val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();

    T work[COUNT];
    _waveOpCopy<INTF, T, COUNT>(work, val);
    _waveOpSetInitial<INTF, T, COUNT>(val, val);

    if (offsetSize > 0)
    {
        // For transmitted value we will do it inclusively with this lanes value
        // For the result we do not include the lanes value. This means an extra op for each
        // iteration but means we don't need to have a divide at the end and also removes overflow
        // issues in that scenario.
        for (int i = 1; i < offsetSize; i += i)
        {
            for (size_t j = 0; j < COUNT; ++j)
            {
                const T readVal = __shfl_up_sync(mask, work[j], i, offsetSize);
                if (laneId >= i)
                {
                    work[j] = INTF::doOp(work[j], readVal);
                    val[j] = INTF::doOp(val[j], readVal);
                }
            }
        }
    }
    else
    {
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;

                for (size_t j = 0; j < COUNT; ++j)
                {
                    // Broadcast (can also broadcast to self)
                    const T readValue = __shfl_sync(mask, work[j], srcLane);
                    // Only accumulate if srcLane is less than this lane
                    if (srcLane < laneId)
                    {
                        val[j] = INTF::doOp(val[j], readValue);
                    }
                }
                remaining &= ~laneBit;
            }
        }
    }
}

template<typename T>
__inline__ __device__ T _wavePrefixProduct(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpMul<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixSum(WarpMask mask, T val)
{
    return _wavePrefixInvertableScalar<WaveOpAdd<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixXor(WarpMask mask, T val)
{
    return _wavePrefixInvertableScalar<WaveOpXor<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixOr(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpOr<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixAnd(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpAnd<T>, T>(mask, val);
}


template<typename T>
__inline__ __device__ T _wavePrefixProductMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixInvertableMultiple<WaveOpMul<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixSumMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixInvertableMultiple<WaveOpAdd<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixXorMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixInvertableMultiple<WaveOpXor<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixOrMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixMultiple<WaveOpOr<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixAndMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixMultiple<WaveOpAnd<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ uint4 _waveMatchScalar(WarpMask mask, T val)
{
    int pred;
    return make_uint4(__match_all_sync(mask, val, &pred), 0, 0, 0);
}

template<typename T>
__inline__ __device__ uint4 _waveMatchMultiple(WarpMask mask, const T& inVal)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    int pred;
    const ElemType* src = (const ElemType*)&inVal;
    uint matchBits = 0xffffffff;
    for (size_t i = 0; i < count && matchBits; ++i)
    {
        matchBits = matchBits & __match_all_sync(mask, src[i], &pred);
    }
    return make_uint4(matchBits, 0, 0, 0);
}

__device__ uint getAt(dim3 a, int b)
{
    SLANG_PRELUDE_ASSERT(b >= 0 && b < 3);
    return (&a.x)[b];
}
__device__ uint3 operator*(uint3 a, dim3 b)
{
    uint3 r;
    r.x = a.x * b.x;
    r.y = a.y * b.y;
    r.z = a.z * b.z;
    return r;
}

template<typename TResult, typename TInput>
__inline__ __device__ TResult slang_bit_cast(TInput val)
{
    return *(TResult*)(&val);
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */


/* Type that defines the uniform entry point params. The actual content of this type is dependent on
the entry point parameters, and can be found via reflection or defined such that it matches the
shader appropriately.
*/
struct UniformEntryPointParams;
struct UniformState;

// ---------------------- OptiX Ray Payload --------------------------------------
#ifdef SLANG_CUDA_ENABLE_OPTIX
struct RayDesc
{
    float3 Origin;
    float TMin;
    float3 Direction;
    float TMax;
};

static __forceinline__ __device__ void* unpackOptiXRayPayloadPointer(uint32_t i0, uint32_t i1)
{
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __forceinline__ __device__ void packOptiXRayPayloadPointer(
    void* ptr,
    uint32_t& i0,
    uint32_t& i1)
{
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ void* getOptiXRayPayloadPtr()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return unpackOptiXRayPayloadPointer(u0, u1);
}

template<typename T>
__forceinline__ __device__ void* traceOptiXRay(
    OptixTraversableHandle AccelerationStructure,
    uint32_t RayFlags,
    uint32_t InstanceInclusionMask,
    uint32_t RayContributionToHitGroupIndex,
    uint32_t MultiplierForGeometryContributionToHitGroupIndex,
    uint32_t MissShaderIndex,
    RayDesc Ray,
    T* Payload)
{
    uint32_t r0, r1;
    packOptiXRayPayloadPointer((void*)Payload, r0, r1);
    optixTrace(
        AccelerationStructure,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        Ray.TMax,
        0.f, /* Time for motion blur, currently unsupported in slang */
        InstanceInclusionMask,
        RayFlags,
        RayContributionToHitGroupIndex,
        MultiplierForGeometryContributionToHitGroupIndex,
        MissShaderIndex,
        r0,
        r1);
}

#endif

static const int kSlangTorchTensorMaxDim = 5;

// TensorView
struct TensorView
{
    uint8_t* data;
    uint32_t strides[kSlangTorchTensorMaxDim];
    uint32_t sizes[kSlangTorchTensorMaxDim];
    uint32_t dimensionCount;

    template<typename T>
    __device__ T* data_ptr()
    {
        return reinterpret_cast<T*>(data);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint32_t index)
    {
        uint64_t offset = strides[0] * index;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint2 index)
    {
        uint64_t offset = strides[0] * index.x + strides[1] * index.y;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint3 index)
    {
        uint64_t offset = strides[0] * index.x + strides[1] * index.y + strides[2] * index.z;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint4 index)
    {
        uint64_t offset = strides[0] * index.x + strides[1] * index.y + strides[2] * index.z +
                          strides[3] * index.w;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T, unsigned int N>
    __device__ T* data_ptr_at(uint index[N])
    {
        uint64_t offset = 0;
        for (unsigned int i = 0; i < N; ++i)
        {
            offset += strides[i] * index[i];
        }
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T& load(uint32_t x)
    {
        return *reinterpret_cast<T*>(data + strides[0] * x);
    }
    template<typename T>
    __device__ T& load(uint32_t x, uint32_t y)
    {
        return *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y);
    }
    template<typename T>
    __device__ T& load(uint2 index)
    {
        return *reinterpret_cast<T*>(data + strides[0] * index.x + strides[1] * index.y);
    }
    template<typename T>
    __device__ T& load(uint32_t x, uint32_t y, uint32_t z)
    {
        return *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y + strides[2] * z);
    }
    template<typename T>
    __device__ T& load(uint3 index)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z);
    }
    template<typename T>
    __device__ T& load(uint32_t x, uint32_t y, uint32_t z, uint32_t w)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * x + strides[1] * y + strides[2] * z + strides[3] * w);
    }
    template<typename T>
    __device__ T& load(uint4 index)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z +
            strides[3] * index.w);
    }
    template<typename T>
    __device__ T& load(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t i4)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * i0 + strides[1] * i1 + strides[2] * i2 + strides[3] * i3 +
            strides[4] * i4);
    }

    // Generic version of load
    template<typename T, unsigned int N>
    __device__ T& load(uint index[N])
    {
        uint64_t offset = 0;
        for (unsigned int i = 0; i < N; ++i)
        {
            offset += strides[i] * index[i];
        }
        return *reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ void store(uint32_t x, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * x) = val;
    }
    template<typename T>
    __device__ void store(uint32_t x, uint32_t y, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y) = val;
    }
    template<typename T>
    __device__ void store(uint2 index, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * index.x + strides[1] * index.y) = val;
    }
    template<typename T>
    __device__ void store(uint32_t x, uint32_t y, uint32_t z, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y + strides[2] * z) = val;
    }
    template<typename T>
    __device__ void store(uint3 index, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z) = val;
    }
    template<typename T>
    __device__ void store(uint32_t x, uint32_t y, uint32_t z, uint32_t w, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * x + strides[1] * y + strides[2] * z + strides[3] * w) = val;
    }
    template<typename T>
    __device__ void store(uint4 index, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z +
            strides[3] * index.w) = val;
    }
    template<typename T>
    __device__ void store(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t i4, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * i0 + strides[1] * i1 + strides[2] * i2 + strides[3] * i3 +
            strides[4] * i4) = val;
    }

    // Generic version
    template<typename T, unsigned int N>
    __device__ void store(uint index[N], T val)
    {
        uint64_t offset = 0;
        for (unsigned int i = 0; i < N; ++i)
        {
            offset += strides[i] * index[i];
        }
        *reinterpret_cast<T*>(data + offset) = val;
    }
};


#line 7891 "hlsl.meta.slang"
struct DiffPair_float_0
{
    float primal_0;
    float differential_0;
};


#line 1935 "diff.meta.slang"
__device__ void _d_max_0(DiffPair_float_0 * dpx_0, DiffPair_float_0 * dpy_0, float dOut_0)
{
    DiffPair_float_0 _S1 = *dpx_0;

#line 1937
    float _S2;

#line 1937
    if((*dpx_0).primal_0 > (*dpy_0).primal_0)
    {

#line 1937
        _S2 = dOut_0;

#line 1937
    }
    else
    {

#line 1937
        _S2 = 0.0f;

#line 1937
    }

#line 1937
    dpx_0->primal_0 = _S1.primal_0;

#line 1937
    dpx_0->differential_0 = _S2;
    DiffPair_float_0 _S3 = *dpy_0;

#line 1938
    if((*dpy_0).primal_0 > _S1.primal_0)
    {

#line 1938
        _S2 = dOut_0;

#line 1938
    }
    else
    {

#line 1938
        _S2 = 0.0f;

#line 1938
    }

#line 1938
    dpy_0->primal_0 = _S3.primal_0;

#line 1938
    dpy_0->differential_0 = _S2;
    return;
}


#line 1 "token paste"
__device__ void _d_sqrt_0(DiffPair_float_0 * dpx_1, float dOut_1)
{

#line 1719 "diff.meta.slang"
    float _S4 = 0.5f / (F32_sqrt(((F32_max((1.00000001168609742e-07f), ((*dpx_1).primal_0)))))) * dOut_1;

#line 1719
    dpx_1->primal_0 = (*dpx_1).primal_0;

#line 1719
    dpx_1->differential_0 = _S4;



    return;
}


#line 1723
struct DiffPair_vectorx3Cfloatx2C3x3E_0
{
    float3  primal_0;
    float3  differential_0;
};


#line 1443
__device__ void _d_dot_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpx_2, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpy_1, float dOut_2)
{
    float3  x_d_result_0;



    *&((&x_d_result_0)->x) = (*dpy_1).primal_0.x * dOut_2;

#line 1445
    float3  y_d_result_0;

#line 1450
    *&((&y_d_result_0)->x) = (*dpx_2).primal_0.x * dOut_2;

#line 1449
    *&((&x_d_result_0)->y) = (*dpy_1).primal_0.y * dOut_2;
    *&((&y_d_result_0)->y) = (*dpx_2).primal_0.y * dOut_2;

#line 1449
    *&((&x_d_result_0)->z) = (*dpy_1).primal_0.z * dOut_2;
    *&((&y_d_result_0)->z) = (*dpx_2).primal_0.z * dOut_2;

#line 1450
    dpx_2->primal_0 = (*dpx_2).primal_0;

#line 1450
    dpx_2->differential_0 = x_d_result_0;

#line 1450
    dpy_1->primal_0 = (*dpy_1).primal_0;

#line 1450
    dpy_1->differential_0 = y_d_result_0;



    return;
}


#line 7891 "hlsl.meta.slang"
__device__ float dot_0(float3  x_0, float3  y_0)
{

#line 7891
    int i_0 = int(0);

#line 7891
    float result_0 = 0.0f;

#line 7904
    for(;;)
    {

#line 7904
        if(i_0 < int(3))
        {
        }
        else
        {

#line 7904
            break;
        }

#line 7905
        float result_1 = result_0 + _slang_vector_get_element(x_0, i_0) * _slang_vector_get_element(y_0, i_0);

#line 7904
        i_0 = i_0 + int(1);

#line 7904
        result_0 = result_1;

#line 7904
    }

    return result_0;
}


#line 9729
__device__ float length_0(float3  x_1)
{

#line 9741
    return (F32_sqrt((dot_0(x_1, x_1))));
}


#line 11211
__device__ float3  normalize_0(float3  x_2)
{

#line 11223
    return x_2 / make_float3 (length_0(x_2));
}


#line 1645 "diff.meta.slang"
__device__ void s_bwd_prop_sqrt_0(DiffPair_float_0 * _S5, float _S6)
{

#line 1645
    _d_sqrt_0(_S5, _S6);

#line 1645
    return;
}


#line 2092
__device__ void s_bwd_prop_length_impl_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpx_3, float _s_dOut_0)
{

#line 2092
    float _S7 = (*dpx_3).primal_0.x;

#line 2092
    float _S8 = (*dpx_3).primal_0.y;

#line 2092
    float _S9 = (*dpx_3).primal_0.z;

#line 2099
    DiffPair_float_0 _S10;

#line 2099
    (&_S10)->primal_0 = _S7 * _S7 + _S8 * _S8 + _S9 * _S9;

#line 2099
    (&_S10)->differential_0 = 0.0f;

#line 2099
    s_bwd_prop_sqrt_0(&_S10, _s_dOut_0);

#line 2099
    float _S11 = (*dpx_3).primal_0.z * _S10.differential_0;

#line 958 "core.meta.slang"
    float _S12 = _S11 + _S11;

#line 958
    float _S13 = (*dpx_3).primal_0.y * _S10.differential_0;

#line 958
    float _S14 = _S13 + _S13;

#line 958
    float _S15 = (*dpx_3).primal_0.x * _S10.differential_0;

#line 958
    float _S16 = _S15 + _S15;

#line 958
    float3  _S17 = make_float3 (0.0f);

#line 958
    *&((&_S17)->z) = _S12;

#line 958
    *&((&_S17)->y) = _S14;

#line 958
    *&((&_S17)->x) = _S16;

#line 958
    dpx_3->primal_0 = (*dpx_3).primal_0;

#line 958
    dpx_3->differential_0 = _S17;

#line 2092 "diff.meta.slang"
    return;
}


#line 2092
__device__ void s_bwd_length_impl_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S18, float _S19)
{

#line 2092
    s_bwd_prop_length_impl_0(_S18, _S19);

#line 2092
    return;
}


#line 2154
__device__ void s_bwd_prop_normalize_impl_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpx_4, float3  _s_dOut_1)
{
    float _S20 = length_0((*dpx_4).primal_0);
    float3  _S21 = (*dpx_4).primal_0 * _s_dOut_1;

#line 2157
    float3  _S22 = make_float3 (1.0f / _S20) * _s_dOut_1;

#line 2157
    float _S23 = - ((_S21.x + _S21.y + _S21.z) / (_S20 * _S20));

#line 2156
    float3  _S24 = make_float3 (0.0f);

#line 2156
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S25;

#line 2156
    (&_S25)->primal_0 = (*dpx_4).primal_0;

#line 2156
    (&_S25)->differential_0 = _S24;

#line 2156
    s_bwd_length_impl_0(&_S25, _S23);

#line 2156
    float3  _S26 = _S22 + _S25.differential_0;

#line 2156
    dpx_4->primal_0 = (*dpx_4).primal_0;

#line 2156
    dpx_4->differential_0 = _S26;

#line 2154
    return;
}


#line 2154
__device__ void s_bwd_normalize_impl_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S27, float3  _S28)
{

#line 2154
    s_bwd_prop_normalize_impl_0(_S27, _S28);

#line 2154
    return;
}


#line 70 "/rhome/jschmidt/projects/goliath/ca_code/slang/utils.slang"
__device__ void bwd_safeNormalize_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * v_0, float3  d_out_0)
{

    if((F32_sqrt(((*v_0).primal_0.x * (*v_0).primal_0.x + (*v_0).primal_0.y * (*v_0).primal_0.y + (*v_0).primal_0.z * (*v_0).primal_0.z))) > 0.0f)
    {
        s_bwd_normalize_impl_0(v_0, d_out_0);

#line 73
    }



    return;
}


__device__ float3  safeNormalize_0(float3  v_1)
{
    float _S29 = v_1.x;

#line 83
    float _S30 = v_1.y;

#line 83
    float _S31 = v_1.z;

#line 83
    float l_0 = (F32_sqrt((_S29 * _S29 + _S30 * _S30 + _S31 * _S31)));

#line 83
    float3  _S32;
    if(l_0 > 0.0f)
    {

#line 84
        _S32 = v_1 / make_float3 (l_0);

#line 84
    }
    else
    {

#line 84
        _S32 = make_float3 (0.0f);

#line 84
    }

#line 84
    return _S32;
}


#line 4 "/rhome/jschmidt/projects/goliath/ca_code/slang/disney.slang"
__device__ float lambert_0(float3  nrm_0, float3  wi_0)
{
    return (F32_max((dot_0(nrm_0, wi_0) / 3.14159274101257324f), (0.0f)));
}


#line 2007 "diff.meta.slang"
__device__ void _d_clamp_0(DiffPair_float_0 * dpx_5, DiffPair_float_0 * dpMin_0, DiffPair_float_0 * dpMax_0, float dOut_3)
{
    DiffPair_float_0 _S33 = *dpx_5;

#line 2009
    bool _S34;

#line 2009
    if((*dpx_5).primal_0 > (*dpMin_0).primal_0)
    {

#line 2009
        _S34 = (*dpx_5).primal_0 < (*dpMax_0).primal_0;

#line 2009
    }
    else
    {

#line 2009
        _S34 = false;

#line 2009
    }

#line 2009
    float _S35;

#line 2009
    if(_S34)
    {

#line 2009
        _S35 = dOut_3;

#line 2009
    }
    else
    {

#line 2009
        _S35 = 0.0f;

#line 2009
    }

#line 2009
    dpx_5->primal_0 = _S33.primal_0;

#line 2009
    dpx_5->differential_0 = _S35;
    DiffPair_float_0 _S36 = *dpMin_0;

#line 2010
    if(_S33.primal_0 <= (*dpMin_0).primal_0)
    {

#line 2010
        _S35 = dOut_3;

#line 2010
    }
    else
    {

#line 2010
        _S35 = 0.0f;

#line 2010
    }

#line 2010
    dpMin_0->primal_0 = _S36.primal_0;

#line 2010
    dpMin_0->differential_0 = _S35;
    DiffPair_float_0 _S37 = *dpMax_0;

#line 2011
    if((*dpx_5).primal_0 >= (*dpMax_0).primal_0)
    {

#line 2011
        _S35 = dOut_3;

#line 2011
    }
    else
    {

#line 2011
        _S35 = 0.0f;

#line 2011
    }

#line 2011
    dpMax_0->primal_0 = _S37.primal_0;

#line 2011
    dpMax_0->differential_0 = _S35;
    return;
}


#line 7128 "hlsl.meta.slang"
__device__ float clamp_0(float x_3, float minBound_0, float maxBound_0)
{

#line 7140
    return (F32_min(((F32_max((x_3), (minBound_0)))), (maxBound_0)));
}


#line 19 "/rhome/jschmidt/projects/goliath/ca_code/slang/disney.slang"
__device__ float ndfGGX_0(float alphaSqr_0, float cosTheta_0)
{

    float _cosTheta_0 = clamp_0(cosTheta_0, 0.00009999999747379f, 0.99989998340606689f);
    float d_0 = (_cosTheta_0 * alphaSqr_0 - _cosTheta_0) * _cosTheta_0 + 1.0f;
    return alphaSqr_0 / (d_0 * d_0 * 3.14159274101257324f);
}


__device__ float lambdaGGX_0(float alphaSqr_1, float cosTheta_1)
{

    float _cosTheta_1 = clamp_0(cosTheta_1, 0.00009999999747379f, 0.99989998340606689f);
    float cosThetaSqr_0 = _cosTheta_1 * _cosTheta_1;

    return 0.5f * ((F32_sqrt((1.0f + alphaSqr_1 * ((1.0f - cosThetaSqr_0) / cosThetaSqr_0)))) - 1.0f);
}


__device__ float maskingSmithGGXCorrelated_0(float alphaSqr_2, float cosThetaI_0, float cosThetaO_0)
{


    return 1.0f / (1.0f + lambdaGGX_0(alphaSqr_2, cosThetaI_0) + lambdaGGX_0(alphaSqr_2, cosThetaO_0));
}


#line 1896 "diff.meta.slang"
__device__ void _d_pow_0(DiffPair_float_0 * dpx_6, DiffPair_float_0 * dpy_2, float dOut_4)
{

    if((*dpx_6).primal_0 < 9.99999997475242708e-07f)
    {

#line 1899
        dpx_6->primal_0 = (*dpx_6).primal_0;

#line 1899
        dpx_6->differential_0 = 0.0f;

#line 1899
        dpy_2->primal_0 = (*dpy_2).primal_0;

#line 1899
        dpy_2->differential_0 = 0.0f;

#line 1899
    }
    else
    {

#line 1906
        float val_0 = (F32_pow(((*dpx_6).primal_0), ((*dpy_2).primal_0)));

        DiffPair_float_0 _S38 = *dpx_6;

#line 1908
        float _S39 = val_0 * (*dpy_2).primal_0 / (*dpx_6).primal_0 * dOut_4;

#line 1908
        dpx_6->primal_0 = (*dpx_6).primal_0;

#line 1908
        dpx_6->differential_0 = _S39;

#line 1908
        float _S40 = val_0 * (F32_log((_S38.primal_0))) * dOut_4;

#line 1908
        dpy_2->primal_0 = (*dpy_2).primal_0;

#line 1908
        dpy_2->differential_0 = _S40;

#line 1899
    }

#line 1914
    return;
}


#line 10 "/rhome/jschmidt/projects/goliath/ca_code/slang/disney.slang"
__device__ float3  fresnelSchlick_0(float3  f0_0, float3  f90_0, float cosTheta_2)
{


    float scale_0 = (F32_pow((1.0f - clamp_0(cosTheta_2, 0.00009999999747379f, 0.99989998340606689f)), (5.0f)));
    return f0_0 * make_float3 (1.0f - scale_0) + f90_0 * make_float3 (scale_0);
}


#line 46
__device__ float3  pbrSpecular_0(float3  col_0, float3  nrm_1, float3  wo_0, float3  wi_1, float alpha_0, float min_roughness_0)
{
    float _alpha_0 = clamp_0(alpha_0, min_roughness_0 * min_roughness_0, 1.0f);
    float alphaSqr_3 = _alpha_0 * _alpha_0;

    float woDotN_0 = dot_0(wo_0, nrm_1);
    float wiDotN_0 = dot_0(wi_1, nrm_1);



    float3  _S41 = make_float3 (0.0f);

#line 56
    float3  res_0;
    if(woDotN_0 > 0.00009999999747379f & wiDotN_0 > 0.00009999999747379f)
    {
        float3  h_0 = safeNormalize_0(wo_0 + wi_1);

#line 59
        res_0 = fresnelSchlick_0(col_0, make_float3 (1.0f), dot_0(wo_0, h_0)) * make_float3 (ndfGGX_0(alphaSqr_3, dot_0(nrm_1, h_0))) * make_float3 (maskingSmithGGXCorrelated_0(alphaSqr_3, woDotN_0, wiDotN_0)) * make_float3 (0.25f) / make_float3 (woDotN_0);

#line 57
    }
    else
    {

#line 57
        res_0 = _S41;

#line 57
    }

#line 67
    return res_0;
}


#line 113
__device__ float3  pbrBSDF_0(float3  kd_0, float3  arm_0, float3  pos_0, float3  nrm_2, float3  light_intensity_0, float3  view_pos_0, float3  light_pos_0, float min_roughness_1)
{

#line 123
    float3  wi_2 = safeNormalize_0(light_pos_0 - pos_0);

    float _S42 = arm_0.y;

    float metallic_0 = arm_0.z;
    float _S43 = 1.0f - metallic_0;

#line 137
    return (kd_0 * make_float3 (_S43) * make_float3 (lambert_0(nrm_2, wi_2)) + make_float3 (arm_0.x) * pbrSpecular_0(make_float3 (0.03999999910593033f * _S43) + kd_0 * make_float3 (metallic_0), nrm_2, safeNormalize_0(view_pos_0 - pos_0), wi_2, _S42 * _S42, min_roughness_1)) * light_intensity_0;
}


#line 144
__global__ void pbr_bn_fwd_kernel(TensorView kd_1, TensorView arm_1, TensorView pos_1, TensorView nrm_3, TensorView view_pos_1, TensorView light_pos_1, TensorView light_intensity_1, float min_roughness_2, TensorView output_0)
{

#line 154
    uint3  _S44 = ((blockIdx)) * ((blockDim)) + ((threadIdx));
    uint _S45 = _S44.x;

#line 155
    uint _S46 = ((output_0).sizes[(1U)]);

#line 155
    bool _S47;

#line 155
    if(_S45 > _S46)
    {

#line 155
        _S47 = true;

#line 155
    }
    else
    {

#line 155
        uint _S48 = _S44.y;

#line 155
        uint _S49 = ((output_0).sizes[(0U)]);

#line 155
        _S47 = _S48 > _S49;

#line 155
    }

#line 155
    if(_S47)
    {
        return;
    }

    uint _S50 = _S44.y;
    float3  _S51 = ((view_pos_1).load<float3>((_S50)));

    uint _S52 = ((light_pos_1).sizes[(1U)]);
    (output_0).store<float3 >((_S50), (_S45), (make_float3 (0.0f)));

#line 164
    uint i_1 = 0U;

    for(;;)
    {

#line 166
        if(i_1 < _S52)
        {
        }
        else
        {

#line 166
            break;
        }
        float3  _S53 = ((light_pos_1).load<float3>((_S50), (i_1)));
        float3  _S54 = ((light_intensity_1).load<float3>((_S50), (i_1)));
        if(!(length_0(_S54) > 0.0f))
        {
            i_1 = i_1 + 1U;

#line 166
            continue;
        }

#line 175
        float3  _S55 = ((kd_1).load<float3>((_S50), (_S45)));

#line 175
        float3  _S56 = ((arm_1).load<float3>((_S50), (_S45)));

#line 175
        float3  _S57 = ((pos_1).load<float3>((_S50), (_S45)));

#line 175
        float3  _S58 = ((nrm_3).load<float3>((_S50), (_S45)));

#line 175
        float3  _S59 = pbrBSDF_0(_S55, _S56, _S57, _S58, _S54, _S51, _S53, min_roughness_2);
        float3  _S60 = ((output_0).load<float3>((_S50), (_S45)));

#line 176
        (output_0).store<float3 >((_S50), (_S45), (_S60 + _S59));

#line 166
        i_1 = i_1 + 1U;

#line 166
    }

#line 178
    return;
}


#line 9491 "hlsl.meta.slang"
__device__ bool3  isfinite_0(float3  x_4)
{

#line 5504
    bool3  result_2;

#line 5504
    int i_2 = int(0);

#line 5504
    for(;;)
    {

#line 5504
        if(i_2 < int(3))
        {
        }
        else
        {

#line 5504
            break;
        }

#line 5504
        *_slang_vector_get_element_ptr(&result_2, i_2) = (F32_isfinite((_slang_vector_get_element(x_4, i_2))));

#line 5504
        i_2 = i_2 + int(1);

#line 5504
    }

#line 5504
    return result_2;
}


#line 10450
__device__ float3  min_0(float3  x_5, float3  y_1)
{

#line 5510
    float3  result_3;

#line 5510
    int i_3 = int(0);

#line 5510
    for(;;)
    {

#line 5510
        if(i_3 < int(3))
        {
        }
        else
        {

#line 5510
            break;
        }

#line 5510
        *_slang_vector_get_element_ptr(&result_3, i_3) = (F32_min((_slang_vector_get_element(x_5, i_3)), (_slang_vector_get_element(y_1, i_3))));

#line 5510
        i_3 = i_3 + int(1);

#line 5510
    }

#line 5510
    return result_3;
}


#line 10224
__device__ float3  max_0(float3  x_6, float3  y_2)
{

#line 5510
    float3  result_4;

#line 5510
    int i_4 = int(0);

#line 5510
    for(;;)
    {

#line 5510
        if(i_4 < int(3))
        {
        }
        else
        {

#line 5510
            break;
        }

#line 5510
        *_slang_vector_get_element_ptr(&result_4, i_4) = (F32_max((_slang_vector_get_element(x_6, i_4)), (_slang_vector_get_element(y_2, i_4))));

#line 5510
        i_4 = i_4 + int(1);

#line 5510
    }

#line 5510
    return result_4;
}


#line 7147
__device__ float3  clamp_1(float3  x_7, float3  minBound_1, float3  maxBound_1)
{

#line 7159
    return min_0(max_0(x_7, minBound_1), maxBound_1);
}


#line 181 "/rhome/jschmidt/projects/goliath/ca_code/slang/disney.slang"
__device__ float3  s_primal_ctx_safeNormalize_0(float3  _S61)
{

#line 181
    return safeNormalize_0(_S61);
}


#line 181
__device__ float s_primal_ctx_dot_0(float3  _S62, float3  _S63)
{

#line 181
    return dot_0(_S62, _S63);
}


#line 181
__device__ float s_primal_ctx_max_0(float _S64, float _S65)
{

#line 181
    return (F32_max((_S64), (_S65)));
}


#line 181
__device__ float s_primal_ctx_lambert_0(float3  dpnrm_0, float3  dpwi_0)
{

#line 4
    return s_primal_ctx_max_0(s_primal_ctx_dot_0(dpnrm_0, dpwi_0) / 3.14159274101257324f, 0.0f);
}


#line 4
__device__ float s_primal_ctx_clamp_0(float _S66, float _S67, float _S68)
{

#line 4
    return clamp_0(_S66, _S67, _S68);
}


#line 4
__device__ float s_primal_ctx_ndfGGX_0(float dpalphaSqr_0, float dpcosTheta_0)
{

#line 19
    float _S69 = s_primal_ctx_clamp_0(dpcosTheta_0, 0.00009999999747379f, 0.99989998340606689f);



    float d_1 = (_S69 * dpalphaSqr_0 - _S69) * _S69 + 1.0f;

#line 23
    return dpalphaSqr_0 / (d_1 * d_1 * 3.14159274101257324f);
}


#line 23
__device__ float s_primal_ctx_sqrt_0(float _S70)
{

#line 23
    return (F32_sqrt((_S70)));
}


#line 23
__device__ float s_primal_ctx_lambdaGGX_0(float dpalphaSqr_1, float dpcosTheta_1)
{



    float _S71 = s_primal_ctx_clamp_0(dpcosTheta_1, 0.00009999999747379f, 0.99989998340606689f);



    float cosThetaSqr_1 = _S71 * _S71;

#line 32
    return 0.5f * (s_primal_ctx_sqrt_0(1.0f + dpalphaSqr_1 * ((1.0f - cosThetaSqr_1) / cosThetaSqr_1)) - 1.0f);
}


#line 32
__device__ float s_primal_ctx_maskingSmithGGXCorrelated_0(float dpalphaSqr_2, float dpcosThetaI_0, float dpcosThetaO_0)
{

#line 38
    return 1.0f / (1.0f + s_primal_ctx_lambdaGGX_0(dpalphaSqr_2, dpcosThetaI_0) + s_primal_ctx_lambdaGGX_0(dpalphaSqr_2, dpcosThetaO_0));
}


#line 38
__device__ float s_primal_ctx_pow_0(float _S72, float _S73)
{

#line 38
    return (F32_pow((_S72), (_S73)));
}


#line 38
__device__ float3  s_primal_ctx_fresnelSchlick_0(float3  dpf0_0, float3  dpf90_0, float dpcosTheta_2)
{

#line 10
    float _S74 = s_primal_ctx_pow_0(1.0f - s_primal_ctx_clamp_0(dpcosTheta_2, 0.00009999999747379f, 0.99989998340606689f), 5.0f);

#line 10
    return dpf0_0 * make_float3 (1.0f - _S74) + dpf90_0 * make_float3 (_S74);
}


#line 10
__device__ float3  s_primal_ctx_pbrSpecular_0(float3  dpcol_0, float3  dpnrm_1, float3  dpwo_0, float3  dpwi_1, float dpalpha_0, float min_roughness_3)
{

#line 46
    float _S75 = s_primal_ctx_clamp_0(dpalpha_0, min_roughness_3 * min_roughness_3, 1.0f);


    float alphaSqr_4 = _S75 * _S75;

#line 49
    float _S76 = s_primal_ctx_dot_0(dpwo_0, dpnrm_1);

#line 49
    float _S77 = s_primal_ctx_dot_0(dpwi_1, dpnrm_1);

#line 56
    float3  _S78 = make_float3 (0.0f);

#line 56
    float3  res_1;
    if(_S76 > 0.00009999999747379f & _S77 > 0.00009999999747379f)
    {

#line 57
        float3  _S79 = s_primal_ctx_safeNormalize_0(dpwo_0 + dpwi_1);

#line 57
        res_1 = s_primal_ctx_fresnelSchlick_0(dpcol_0, make_float3 (1.0f), s_primal_ctx_dot_0(dpwo_0, _S79)) * make_float3 (s_primal_ctx_ndfGGX_0(alphaSqr_4, s_primal_ctx_dot_0(dpnrm_1, _S79))) * make_float3 (s_primal_ctx_maskingSmithGGXCorrelated_0(alphaSqr_4, _S76, _S77)) * make_float3 (0.25f) / make_float3 (_S76);

#line 57
    }
    else
    {

#line 57
        res_1 = _S78;

#line 57
    }

#line 57
    return res_1;
}


#line 57
__device__ void s_bwd_prop_pow_0(DiffPair_float_0 * _S80, DiffPair_float_0 * _S81, float _S82)
{

#line 57
    _d_pow_0(_S80, _S81, _S82);

#line 57
    return;
}


#line 57
__device__ void s_bwd_prop_clamp_0(DiffPair_float_0 * _S83, DiffPair_float_0 * _S84, DiffPair_float_0 * _S85, float _S86)
{

#line 57
    _d_clamp_0(_S83, _S84, _S85, _S86);

#line 57
    return;
}


#line 10
__device__ void s_bwd_prop_fresnelSchlick_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpf0_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpf90_1, DiffPair_float_0 * dpcosTheta_3, float3  _s_dOut_2)
{


    float _S87 = 1.0f - s_primal_ctx_clamp_0((*dpcosTheta_3).primal_0, 0.00009999999747379f, 0.99989998340606689f);

#line 14
    float _S88 = s_primal_ctx_pow_0(_S87, 5.0f);
    float3  _S89 = (*dpf90_1).primal_0 * _s_dOut_2;

#line 15
    float3  _S90 = make_float3 (_S88) * _s_dOut_2;

#line 15
    float3  _S91 = (*dpf0_1).primal_0 * _s_dOut_2;

#line 15
    float3  _S92 = make_float3 (1.0f - _S88) * _s_dOut_2;

#line 14
    float _S93 = - (_S91.x + _S91.y + _S91.z) + _S89.x + _S89.y + _S89.z;

#line 14
    DiffPair_float_0 _S94;

#line 14
    (&_S94)->primal_0 = _S87;

#line 14
    (&_S94)->differential_0 = 0.0f;

#line 14
    DiffPair_float_0 _S95;

#line 14
    (&_S95)->primal_0 = 5.0f;

#line 14
    (&_S95)->differential_0 = 0.0f;

#line 14
    s_bwd_prop_pow_0(&_S94, &_S95, _S93);

#line 14
    float _S96 = - _S94.differential_0;

#line 13
    DiffPair_float_0 _S97;

#line 13
    (&_S97)->primal_0 = (*dpcosTheta_3).primal_0;

#line 13
    (&_S97)->differential_0 = 0.0f;

#line 13
    DiffPair_float_0 _S98;

#line 13
    (&_S98)->primal_0 = 0.00009999999747379f;

#line 13
    (&_S98)->differential_0 = 0.0f;

#line 13
    DiffPair_float_0 _S99;

#line 13
    (&_S99)->primal_0 = 0.99989998340606689f;

#line 13
    (&_S99)->differential_0 = 0.0f;

#line 13
    s_bwd_prop_clamp_0(&_S97, &_S98, &_S99, _S96);

#line 13
    dpcosTheta_3->primal_0 = (*dpcosTheta_3).primal_0;

#line 13
    dpcosTheta_3->differential_0 = _S97.differential_0;

#line 13
    dpf90_1->primal_0 = (*dpf90_1).primal_0;

#line 13
    dpf90_1->differential_0 = _S90;

#line 13
    dpf0_1->primal_0 = (*dpf0_1).primal_0;

#line 13
    dpf0_1->differential_0 = _S92;

#line 10
    return;
}


#line 28
__device__ void s_bwd_prop_lambdaGGX_0(DiffPair_float_0 * dpalphaSqr_3, DiffPair_float_0 * dpcosTheta_4, float _s_dOut_3)
{

#line 28
    float _S100 = s_primal_ctx_clamp_0((*dpcosTheta_4).primal_0, 0.00009999999747379f, 0.99989998340606689f);



    float cosThetaSqr_2 = _S100 * _S100;
    float _S101 = 1.0f - cosThetaSqr_2;

#line 33
    float tanThetaSqr_0 = _S101 / cosThetaSqr_2;

#line 33
    float _S102 = cosThetaSqr_2 * cosThetaSqr_2;
    float _S103 = 0.5f * _s_dOut_3;

#line 34
    DiffPair_float_0 _S104;

#line 34
    (&_S104)->primal_0 = 1.0f + (*dpalphaSqr_3).primal_0 * tanThetaSqr_0;

#line 34
    (&_S104)->differential_0 = 0.0f;

#line 34
    s_bwd_prop_sqrt_0(&_S104, _S103);

#line 34
    float _S105 = tanThetaSqr_0 * _S104.differential_0;

#line 33
    float _S106 = (*dpalphaSqr_3).primal_0 * _S104.differential_0 / _S102;

#line 32
    float _S107 = _S100 * (_S101 * - _S106 + - (cosThetaSqr_2 * _S106));

#line 31
    float _S108 = _S107 + _S107;

#line 31
    DiffPair_float_0 _S109;

#line 31
    (&_S109)->primal_0 = (*dpcosTheta_4).primal_0;

#line 31
    (&_S109)->differential_0 = 0.0f;

#line 31
    DiffPair_float_0 _S110;

#line 31
    (&_S110)->primal_0 = 0.00009999999747379f;

#line 31
    (&_S110)->differential_0 = 0.0f;

#line 31
    DiffPair_float_0 _S111;

#line 31
    (&_S111)->primal_0 = 0.99989998340606689f;

#line 31
    (&_S111)->differential_0 = 0.0f;

#line 31
    s_bwd_prop_clamp_0(&_S109, &_S110, &_S111, _S108);

#line 31
    dpcosTheta_4->primal_0 = (*dpcosTheta_4).primal_0;

#line 31
    dpcosTheta_4->differential_0 = _S109.differential_0;

#line 31
    dpalphaSqr_3->primal_0 = (*dpalphaSqr_3).primal_0;

#line 31
    dpalphaSqr_3->differential_0 = _S105;

#line 28
    return;
}


#line 38
__device__ void s_bwd_prop_maskingSmithGGXCorrelated_0(DiffPair_float_0 * dpalphaSqr_4, DiffPair_float_0 * dpcosThetaI_1, DiffPair_float_0 * dpcosThetaO_1, float _s_dOut_4)
{


    float _S112 = 1.0f + s_primal_ctx_lambdaGGX_0((*dpalphaSqr_4).primal_0, (*dpcosThetaI_1).primal_0) + s_primal_ctx_lambdaGGX_0((*dpalphaSqr_4).primal_0, (*dpcosThetaO_1).primal_0);

#line 42
    float _S113 = - (_s_dOut_4 / (_S112 * _S112));

#line 41
    DiffPair_float_0 _S114;

#line 41
    (&_S114)->primal_0 = (*dpalphaSqr_4).primal_0;

#line 41
    (&_S114)->differential_0 = 0.0f;

#line 41
    DiffPair_float_0 _S115;

#line 41
    (&_S115)->primal_0 = (*dpcosThetaO_1).primal_0;

#line 41
    (&_S115)->differential_0 = 0.0f;

#line 41
    s_bwd_prop_lambdaGGX_0(&_S114, &_S115, _S113);

#line 40
    DiffPair_float_0 _S116;

#line 40
    (&_S116)->primal_0 = (*dpalphaSqr_4).primal_0;

#line 40
    (&_S116)->differential_0 = 0.0f;

#line 40
    DiffPair_float_0 _S117;

#line 40
    (&_S117)->primal_0 = (*dpcosThetaI_1).primal_0;

#line 40
    (&_S117)->differential_0 = 0.0f;

#line 40
    s_bwd_prop_lambdaGGX_0(&_S116, &_S117, _S113);

#line 40
    dpcosThetaO_1->primal_0 = (*dpcosThetaO_1).primal_0;

#line 40
    dpcosThetaO_1->differential_0 = _S115.differential_0;

#line 40
    dpcosThetaI_1->primal_0 = (*dpcosThetaI_1).primal_0;

#line 40
    dpcosThetaI_1->differential_0 = _S117.differential_0;

#line 958 "core.meta.slang"
    float _S118 = _S114.differential_0 + _S116.differential_0;

#line 958
    dpalphaSqr_4->primal_0 = (*dpalphaSqr_4).primal_0;

#line 958
    dpalphaSqr_4->differential_0 = _S118;

#line 38 "/rhome/jschmidt/projects/goliath/ca_code/slang/disney.slang"
    return;
}


#line 19
__device__ void s_bwd_prop_ndfGGX_0(DiffPair_float_0 * dpalphaSqr_5, DiffPair_float_0 * dpcosTheta_5, float _s_dOut_5)
{

#line 19
    float _S119 = s_primal_ctx_clamp_0((*dpcosTheta_5).primal_0, 0.00009999999747379f, 0.99989998340606689f);



    float _S120 = _S119 * (*dpalphaSqr_5).primal_0 - _S119;

#line 23
    float d_2 = _S120 * _S119 + 1.0f;
    float _S121 = d_2 * d_2 * 3.14159274101257324f;

#line 24
    float _S122 = _s_dOut_5 / (_S121 * _S121);

#line 24
    float _S123 = _S121 * _S122;

#line 24
    float _S124 = d_2 * (3.14159274101257324f * ((*dpalphaSqr_5).primal_0 * - _S122));

#line 23
    float _S125 = _S124 + _S124;

#line 23
    float _S126 = _S119 * _S125;

#line 23
    float _S127 = _S119 * _S126;

#line 22
    float _S128 = _S120 * _S125 + - _S126 + (*dpalphaSqr_5).primal_0 * _S126;

#line 22
    DiffPair_float_0 _S129;

#line 22
    (&_S129)->primal_0 = (*dpcosTheta_5).primal_0;

#line 22
    (&_S129)->differential_0 = 0.0f;

#line 22
    DiffPair_float_0 _S130;

#line 22
    (&_S130)->primal_0 = 0.00009999999747379f;

#line 22
    (&_S130)->differential_0 = 0.0f;

#line 22
    DiffPair_float_0 _S131;

#line 22
    (&_S131)->primal_0 = 0.99989998340606689f;

#line 22
    (&_S131)->differential_0 = 0.0f;

#line 22
    s_bwd_prop_clamp_0(&_S129, &_S130, &_S131, _S128);

#line 22
    dpcosTheta_5->primal_0 = (*dpcosTheta_5).primal_0;

#line 22
    dpcosTheta_5->differential_0 = _S129.differential_0;

#line 958 "core.meta.slang"
    float _S132 = _S123 + _S127;

#line 958
    dpalphaSqr_5->primal_0 = (*dpalphaSqr_5).primal_0;

#line 958
    dpalphaSqr_5->differential_0 = _S132;

#line 19 "/rhome/jschmidt/projects/goliath/ca_code/slang/disney.slang"
    return;
}


#line 19
__device__ void s_bwd_prop_dot_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S133, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S134, float _S135)
{

#line 19
    _d_dot_0(_S133, _S134, _S135);

#line 19
    return;
}


#line 19
__device__ void s_bwd_prop_safeNormalize_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S136, float3  _S137)
{

#line 19
    bwd_safeNormalize_0(_S136, _S137);

#line 19
    return;
}


#line 46
__device__ void s_bwd_prop_pbrSpecular_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpcol_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpnrm_2, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpwo_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpwi_2, DiffPair_float_0 * dpalpha_1, float min_roughness_4, float3  _s_dOut_6)
{

#line 46
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S138 = *dpcol_1;

#line 46
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S139 = *dpnrm_2;

#line 46
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S140 = *dpwo_1;

#line 46
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S141 = *dpwi_2;

#line 46
    DiffPair_float_0 _S142 = *dpalpha_1;

#line 59
    float3  _S143 = make_float3 (0.0f);

#line 48
    float _S144 = min_roughness_4 * min_roughness_4;

#line 48
    float _S145 = s_primal_ctx_clamp_0((*dpalpha_1).primal_0, _S144, 1.0f);
    float alphaSqr_5 = _S145 * _S145;

#line 49
    float _S146 = s_primal_ctx_dot_0((*dpwo_1).primal_0, (*dpnrm_2).primal_0);

#line 65
    float3  _S147 = make_float3 (_S146);

#line 65
    float _S148 = s_primal_ctx_dot_0((*dpwi_2).primal_0, (*dpnrm_2).primal_0);

#line 54
    bool frontfacing_0 = _S146 > 0.00009999999747379f & _S148 > 0.00009999999747379f;

#line 54
    float3  _S149;

#line 54
    float3  _S150;

#line 54
    float3  _S151;

#line 54
    float3  _S152;

#line 54
    float3  _S153;

#line 54
    float3  _S154;

#line 54
    float3  _S155;

#line 54
    float3  _S156;

#line 54
    float3  _S157;

#line 54
    float _S158;

#line 54
    float _S159;


    if(frontfacing_0)
    {
        float3  _S160 = _S140.primal_0 + _S141.primal_0;

#line 59
        float3  _S161 = s_primal_ctx_safeNormalize_0(_S160);

#line 59
        float _S162 = s_primal_ctx_dot_0(_S140.primal_0, _S161);

#line 59
        float _S163 = s_primal_ctx_dot_0(_S139.primal_0, _S161);

#line 59
        float _S164 = s_primal_ctx_ndfGGX_0(alphaSqr_5, _S163);

#line 65
        float3  _S165 = make_float3 (_S164);

#line 65
        float _S166 = s_primal_ctx_maskingSmithGGXCorrelated_0(alphaSqr_5, _S146, _S148);

#line 65
        float3  _S167 = make_float3 (_S166);

#line 64
        float3  _S168 = make_float3 (1.0f);

#line 64
        float3  _S169 = s_primal_ctx_fresnelSchlick_0(_S138.primal_0, _S168, _S162);
        float3  _S170 = _S169 * make_float3 (_S164);

#line 65
        float3  _S171 = _S170 * make_float3 (_S166) * make_float3 (0.25f);

#line 65
        _S149 = make_float3 (_S146 * _S146);

#line 65
        _S150 = _S171;

#line 65
        _S151 = _S170;

#line 65
        _S152 = _S167;

#line 65
        _S153 = _S169;

#line 65
        _S154 = _S165;

#line 65
        _S155 = _S168;

#line 65
        _S158 = _S162;

#line 65
        _S159 = _S163;

#line 65
        _S156 = _S161;

#line 65
        _S157 = _S160;

#line 65
    }
    else
    {

#line 65
        _S149 = _S143;

#line 65
        _S150 = _S143;

#line 65
        _S151 = _S143;

#line 65
        _S152 = _S143;

#line 65
        _S153 = _S143;

#line 65
        _S154 = _S143;

#line 65
        _S155 = _S143;

#line 65
        _S158 = 0.0f;

#line 65
        _S159 = 0.0f;

#line 65
        _S156 = _S143;

#line 65
        _S157 = _S143;

#line 65
    }

#line 65
    float _S172;

#line 65
    if(frontfacing_0)
    {

#line 65
        float3  _S173 = _s_dOut_6 / _S149;

#line 65
        float3  _S174 = _S150 * - _S173;

#line 65
        float3  _S175 = make_float3 (0.25f) * (_S147 * _S173);

#line 65
        float3  _S176 = _S151 * _S175;

#line 65
        float3  _S177 = _S152 * _S175;

#line 65
        float3  _S178 = _S153 * _S177;

#line 65
        float3  _S179 = _S154 * _S177;

#line 64
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S180;

#line 64
        (&_S180)->primal_0 = _S138.primal_0;

#line 64
        (&_S180)->differential_0 = _S143;

#line 64
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S181;

#line 64
        (&_S181)->primal_0 = _S155;

#line 64
        (&_S181)->differential_0 = _S143;

#line 64
        DiffPair_float_0 _S182;

#line 64
        (&_S182)->primal_0 = _S158;

#line 64
        (&_S182)->differential_0 = 0.0f;

#line 64
        s_bwd_prop_fresnelSchlick_0(&_S180, &_S181, &_S182, _S179);

#line 63
        float _S183 = _S176.x + _S176.y + _S176.z;

#line 63
        DiffPair_float_0 _S184;

#line 63
        (&_S184)->primal_0 = alphaSqr_5;

#line 63
        (&_S184)->differential_0 = 0.0f;

#line 63
        DiffPair_float_0 _S185;

#line 63
        (&_S185)->primal_0 = _S146;

#line 63
        (&_S185)->differential_0 = 0.0f;

#line 63
        DiffPair_float_0 _S186;

#line 63
        (&_S186)->primal_0 = _S148;

#line 63
        (&_S186)->differential_0 = 0.0f;

#line 63
        s_bwd_prop_maskingSmithGGXCorrelated_0(&_S184, &_S185, &_S186, _S183);

#line 62
        float _S187 = _S178.x + _S178.y + _S178.z;

#line 62
        DiffPair_float_0 _S188;

#line 62
        (&_S188)->primal_0 = alphaSqr_5;

#line 62
        (&_S188)->differential_0 = 0.0f;

#line 62
        DiffPair_float_0 _S189;

#line 62
        (&_S189)->primal_0 = _S159;

#line 62
        (&_S189)->differential_0 = 0.0f;

#line 62
        s_bwd_prop_ndfGGX_0(&_S188, &_S189, _S187);

#line 61
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S190;

#line 61
        (&_S190)->primal_0 = _S139.primal_0;

#line 61
        (&_S190)->differential_0 = _S143;

#line 61
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S191;

#line 61
        (&_S191)->primal_0 = _S156;

#line 61
        (&_S191)->differential_0 = _S143;

#line 61
        s_bwd_prop_dot_0(&_S190, &_S191, _S189.differential_0);

#line 60
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S192;

#line 60
        (&_S192)->primal_0 = _S140.primal_0;

#line 60
        (&_S192)->differential_0 = _S143;

#line 60
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S193;

#line 60
        (&_S193)->primal_0 = _S156;

#line 60
        (&_S193)->differential_0 = _S143;

#line 60
        s_bwd_prop_dot_0(&_S192, &_S193, _S182.differential_0);

#line 59
        float3  _S194 = _S191.differential_0 + _S193.differential_0;

#line 59
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S195;

#line 59
        (&_S195)->primal_0 = _S157;

#line 59
        (&_S195)->differential_0 = _S143;

#line 59
        s_bwd_prop_safeNormalize_0(&_S195, _S194);

#line 49
        float _S196 = _S184.differential_0 + _S188.differential_0;

#line 49
        float3  _S197 = _S192.differential_0 + _S195.differential_0;

#line 49
        _S158 = _S186.differential_0;

#line 49
        _S149 = _S174;

#line 49
        _S159 = _S185.differential_0;

#line 49
        _S172 = _S196;

#line 49
        _S150 = _S195.differential_0;

#line 49
        _S151 = _S197;

#line 49
        _S152 = _S190.differential_0;

#line 49
        _S153 = _S180.differential_0;

#line 49
    }
    else
    {

#line 49
        _S158 = 0.0f;

#line 49
        _S149 = _S143;

#line 49
        _S159 = 0.0f;

#line 49
        _S172 = 0.0f;

#line 49
        _S150 = _S143;

#line 49
        _S151 = _S143;

#line 49
        _S152 = _S143;

#line 49
        _S153 = _S143;

#line 49
    }


    DiffPair_vectorx3Cfloatx2C3x3E_0 _S198;

#line 52
    (&_S198)->primal_0 = _S141.primal_0;

#line 52
    (&_S198)->differential_0 = _S143;

#line 52
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S199;

#line 52
    (&_S199)->primal_0 = _S139.primal_0;

#line 52
    (&_S199)->differential_0 = _S143;

#line 52
    s_bwd_prop_dot_0(&_S198, &_S199, _S158);

#line 51
    float _S200 = _S149.x + _S149.y + _S149.z + _S159;

#line 51
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S201;

#line 51
    (&_S201)->primal_0 = _S140.primal_0;

#line 51
    (&_S201)->differential_0 = _S143;

#line 51
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S202;

#line 51
    (&_S202)->primal_0 = _S139.primal_0;

#line 51
    (&_S202)->differential_0 = _S143;

#line 51
    s_bwd_prop_dot_0(&_S201, &_S202, _S200);

#line 48
    float _S203 = _S145 * _S172 + _S145 * _S172;

#line 48
    DiffPair_float_0 _S204;

#line 48
    (&_S204)->primal_0 = _S142.primal_0;

#line 48
    (&_S204)->differential_0 = 0.0f;

#line 48
    DiffPair_float_0 _S205;

#line 48
    (&_S205)->primal_0 = _S144;

#line 48
    (&_S205)->differential_0 = 0.0f;

#line 48
    DiffPair_float_0 _S206;

#line 48
    (&_S206)->primal_0 = 1.0f;

#line 48
    (&_S206)->differential_0 = 0.0f;

#line 48
    s_bwd_prop_clamp_0(&_S204, &_S205, &_S206, _S203);

#line 48
    dpalpha_1->primal_0 = (*dpalpha_1).primal_0;

#line 48
    dpalpha_1->differential_0 = _S204.differential_0;

#line 48
    float3  _S207 = _S198.differential_0 + _S150;

#line 48
    dpwi_2->primal_0 = (*dpwi_2).primal_0;

#line 48
    dpwi_2->differential_0 = _S207;

#line 48
    float3  _S208 = _S201.differential_0 + _S151;

#line 48
    dpwo_1->primal_0 = (*dpwo_1).primal_0;

#line 48
    dpwo_1->differential_0 = _S208;

#line 48
    float3  _S209 = _S199.differential_0 + _S202.differential_0 + _S152;

#line 48
    dpnrm_2->primal_0 = (*dpnrm_2).primal_0;

#line 48
    dpnrm_2->differential_0 = _S209;

#line 48
    dpcol_1->primal_0 = (*dpcol_1).primal_0;

#line 48
    dpcol_1->differential_0 = _S153;

#line 46
    return;
}


#line 46
__device__ void s_bwd_prop_max_0(DiffPair_float_0 * _S210, DiffPair_float_0 * _S211, float _S212)
{

#line 46
    _d_max_0(_S210, _S211, _S212);

#line 46
    return;
}


#line 4
__device__ void s_bwd_prop_lambert_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpnrm_3, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpwi_3, float _s_dOut_7)
{
    DiffPair_float_0 _S213;

#line 6
    (&_S213)->primal_0 = s_primal_ctx_dot_0((*dpnrm_3).primal_0, (*dpwi_3).primal_0) / 3.14159274101257324f;

#line 6
    (&_S213)->differential_0 = 0.0f;

#line 6
    DiffPair_float_0 _S214;

#line 6
    (&_S214)->primal_0 = 0.0f;

#line 6
    (&_S214)->differential_0 = 0.0f;

#line 6
    s_bwd_prop_max_0(&_S213, &_S214, _s_dOut_7);

#line 6
    float _S215 = 0.31830987334251404f * _S213.differential_0;

#line 6
    float3  _S216 = make_float3 (0.0f);

#line 6
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S217;

#line 6
    (&_S217)->primal_0 = (*dpnrm_3).primal_0;

#line 6
    (&_S217)->differential_0 = _S216;

#line 6
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S218;

#line 6
    (&_S218)->primal_0 = (*dpwi_3).primal_0;

#line 6
    (&_S218)->differential_0 = _S216;

#line 6
    s_bwd_prop_dot_0(&_S217, &_S218, _S215);

#line 6
    dpwi_3->primal_0 = (*dpwi_3).primal_0;

#line 6
    dpwi_3->differential_0 = _S218.differential_0;

#line 6
    dpnrm_3->primal_0 = (*dpnrm_3).primal_0;

#line 6
    dpnrm_3->differential_0 = _S217.differential_0;

#line 4
    return;
}


#line 113
__device__ void s_bwd_prop_pbrBSDF_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpkd_0, DiffPair_vectorx3Cfloatx2C3x3E_0 * dparm_0, DiffPair_vectorx3Cfloatx2C3x3E_0 * dppos_0, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpnrm_4, DiffPair_vectorx3Cfloatx2C3x3E_0 * dplight_intensity_0, float3  view_pos_2, float3  light_pos_2, float min_roughness_5, float3  _s_dOut_8)
{

#line 122
    float3  _S219 = view_pos_2 - (*dppos_0).primal_0;

#line 122
    float3  _S220 = s_primal_ctx_safeNormalize_0(_S219);
    float3  _S221 = light_pos_2 - (*dppos_0).primal_0;

#line 123
    float3  _S222 = s_primal_ctx_safeNormalize_0(_S221);

    float _S223 = (*dparm_0).primal_0.y;

#line 125
    float alpha_1 = _S223 * _S223;
    float spec_str_0 = (*dparm_0).primal_0.x;
    float metallic_1 = (*dparm_0).primal_0.z;
    float3  _S224 = make_float3 (metallic_1);

#line 128
    float _S225 = 1.0f - metallic_1;
    float3  _S226 = make_float3 (_S225);

#line 128
    float3  spec_col_0 = make_float3 (0.03999999910593033f * _S225) + (*dpkd_0).primal_0 * make_float3 (metallic_1);
    float3  diff_col_0 = (*dpkd_0).primal_0 * make_float3 (_S225);

#line 129
    float _S227 = s_primal_ctx_lambert_0((*dpnrm_4).primal_0, _S222);

    float3  _S228 = make_float3 (_S227);

#line 131
    float3  _S229 = s_primal_ctx_pbrSpecular_0(spec_col_0, (*dpnrm_4).primal_0, _S220, _S222, alpha_1, min_roughness_5);

#line 137
    float3  _S230 = (diff_col_0 * make_float3 (_S227) + make_float3 (spec_str_0) * _S229) * _s_dOut_8;

#line 137
    float3  s_diff_diffuse_T_0 = (*dplight_intensity_0).primal_0 * _s_dOut_8;

#line 132
    float3  _S231 = make_float3 (spec_str_0) * s_diff_diffuse_T_0;

#line 132
    float3  _S232 = _S229 * s_diff_diffuse_T_0;

#line 132
    float3  _S233 = make_float3 (0.0f);

#line 132
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S234;

#line 132
    (&_S234)->primal_0 = spec_col_0;

#line 132
    (&_S234)->differential_0 = _S233;

#line 132
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S235;

#line 132
    (&_S235)->primal_0 = (*dpnrm_4).primal_0;

#line 132
    (&_S235)->differential_0 = _S233;

#line 132
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S236;

#line 132
    (&_S236)->primal_0 = _S220;

#line 132
    (&_S236)->differential_0 = _S233;

#line 132
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S237;

#line 132
    (&_S237)->primal_0 = _S222;

#line 132
    (&_S237)->differential_0 = _S233;

#line 132
    DiffPair_float_0 _S238;

#line 132
    (&_S238)->primal_0 = alpha_1;

#line 132
    (&_S238)->differential_0 = 0.0f;

#line 132
    s_bwd_prop_pbrSpecular_0(&_S234, &_S235, &_S236, &_S237, &_S238, min_roughness_5, _S231);

#line 131
    float3  _S239 = diff_col_0 * s_diff_diffuse_T_0;

#line 131
    float3  s_diff_diff_col_T_0 = _S228 * s_diff_diffuse_T_0;

#line 131
    float _S240 = _S239.x + _S239.y + _S239.z;

#line 131
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S241;

#line 131
    (&_S241)->primal_0 = (*dpnrm_4).primal_0;

#line 131
    (&_S241)->differential_0 = _S233;

#line 131
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S242;

#line 131
    (&_S242)->primal_0 = _S222;

#line 131
    (&_S242)->differential_0 = _S233;

#line 131
    s_bwd_prop_lambert_0(&_S241, &_S242, _S240);

#line 129
    float3  _S243 = (*dpkd_0).primal_0 * s_diff_diff_col_T_0;

#line 129
    float3  _S244 = _S226 * s_diff_diff_col_T_0;

#line 128
    float3  _S245 = (*dpkd_0).primal_0 * _S234.differential_0;

#line 128
    float3  _S246 = _S224 * _S234.differential_0;

#line 127
    float _S247 = - (0.03999999910593033f * (_S234.differential_0.x + _S234.differential_0.y + _S234.differential_0.z) + _S243.x + _S243.y + _S243.z) + _S245.x + _S245.y + _S245.z;

#line 126
    float _S248 = _S232.x + _S232.y + _S232.z;

#line 125
    float _S249 = _S223 * _S238.differential_0;

#line 125
    float _S250 = _S249 + _S249;

#line 123
    float3  _S251 = _S237.differential_0 + _S242.differential_0;

#line 123
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S252;

#line 123
    (&_S252)->primal_0 = _S221;

#line 123
    (&_S252)->differential_0 = _S233;

#line 123
    s_bwd_prop_safeNormalize_0(&_S252, _S251);

#line 123
    float3  _S253 = - _S252.differential_0;

#line 122
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S254;

#line 122
    (&_S254)->primal_0 = _S219;

#line 122
    (&_S254)->differential_0 = _S233;

#line 122
    s_bwd_prop_safeNormalize_0(&_S254, _S236.differential_0);

#line 122
    float3  _S255 = - _S254.differential_0;

#line 122
    dplight_intensity_0->primal_0 = (*dplight_intensity_0).primal_0;

#line 122
    dplight_intensity_0->differential_0 = _S230;

#line 122
    float3  _S256 = _S235.differential_0 + _S241.differential_0;

#line 122
    dpnrm_4->primal_0 = (*dpnrm_4).primal_0;

#line 122
    dpnrm_4->differential_0 = _S256;

#line 122
    float3  _S257 = _S253 + _S255;

#line 122
    dppos_0->primal_0 = (*dppos_0).primal_0;

#line 122
    dppos_0->differential_0 = _S257;

#line 122
    float3  _S258 = make_float3 (_S248, _S250, _S247);

#line 122
    dparm_0->primal_0 = (*dparm_0).primal_0;

#line 122
    dparm_0->differential_0 = _S258;

#line 122
    float3  _S259 = _S244 + _S246;

#line 122
    dpkd_0->primal_0 = (*dpkd_0).primal_0;

#line 122
    dpkd_0->differential_0 = _S259;

#line 113
    return;
}


#line 113
__device__ void s_bwd_pbrBSDF_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S260, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S261, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S262, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S263, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S264, float3  _S265, float3  _S266, float _S267, float3  _S268)
{

#line 120
    s_bwd_prop_pbrBSDF_0(_S260, _S261, _S262, _S263, _S264, _S265, _S266, _S267, _S268);

#line 120
    return;
}


#line 181
__global__ void pbr_bn_bwd_kernel(TensorView kd_2, TensorView arm_2, TensorView pos_2, TensorView nrm_4, TensorView view_pos_3, TensorView light_pos_3, TensorView light_intensity_2, float min_roughness_6, TensorView kd_grad_0, TensorView arm_grad_0, TensorView pos_grad_0, TensorView nrm_grad_0, TensorView light_intensity_grad_0, TensorView grad_out_0)
{

#line 196
    uint3  idx_0 = ((blockIdx)) * ((blockDim)) + ((threadIdx));
    uint _S269 = idx_0.x;

#line 197
    uint _S270 = ((grad_out_0).sizes[(1U)]);

#line 197
    bool _S271;

#line 197
    if(_S269 > _S270)
    {

#line 197
        _S271 = true;

#line 197
    }
    else
    {

#line 197
        uint _S272 = idx_0.y;

#line 197
        uint _S273 = ((grad_out_0).sizes[(0U)]);

#line 197
        _S271 = _S272 > _S273;

#line 197
    }

#line 197
    if(_S271)
    {
        return;
    }

    uint _S274 = idx_0.y;

    float3  _S275 = ((view_pos_3).load<float3>((_S274)));

    uint _S276 = ((light_pos_3).sizes[(1U)]);
    float3  _S277 = make_float3 (0.0f);

#line 212
    float3  _S278 = ((kd_2).load<float3>((_S274), (_S269)));
    float3  _S279 = ((arm_2).load<float3>((_S274), (_S269)));
    float3  _S280 = ((pos_2).load<float3>((_S274), (_S269)));
    float3  _S281 = ((nrm_4).load<float3>((_S274), (_S269)));
    float3  _S282 = ((grad_out_0).load<float3>((_S274), (_S269)));

#line 216
    uint i_5 = 0U;

#line 216
    float3  kd_accum_0 = _S277;

#line 216
    float3  arm_accum_0 = _S277;

#line 216
    float3  pos_accum_0 = _S277;

#line 216
    float3  nrm_accum_0 = _S277;

#line 228
    float3  _S283 = make_float3 (0.0f);

#line 246
    float3  _S284 = make_float3 (-1.0f);

#line 246
    float3  _S285 = make_float3 (1.0f);

#line 218
    for(;;)
    {

#line 218
        if(i_5 < _S276)
        {
        }
        else
        {

#line 218
            break;
        }

        float3  _S286 = ((light_pos_3).load<float3>((_S274), (i_5)));
        float3  _S287 = ((light_intensity_2).load<float3>((_S274), (i_5)));
        if(!(length_0(_S287) > 0.0f))
        {
            i_5 = i_5 + 1U;

#line 218
            continue;
        }

#line 228
        DiffPair_vectorx3Cfloatx2C3x3E_0 dp_kd_0;

#line 228
        (&dp_kd_0)->primal_0 = _S278;

#line 228
        (&dp_kd_0)->differential_0 = _S283;
        DiffPair_vectorx3Cfloatx2C3x3E_0 dp_arm_0;

#line 229
        (&dp_arm_0)->primal_0 = _S279;

#line 229
        (&dp_arm_0)->differential_0 = _S283;
        DiffPair_vectorx3Cfloatx2C3x3E_0 dp_pos_0;

#line 230
        (&dp_pos_0)->primal_0 = _S280;

#line 230
        (&dp_pos_0)->differential_0 = _S283;
        DiffPair_vectorx3Cfloatx2C3x3E_0 dp_nrm_0;

#line 231
        (&dp_nrm_0)->primal_0 = _S281;

#line 231
        (&dp_nrm_0)->differential_0 = _S283;
        DiffPair_vectorx3Cfloatx2C3x3E_0 dp_lint_0;

#line 232
        (&dp_lint_0)->primal_0 = _S287;

#line 232
        (&dp_lint_0)->differential_0 = _S283;

        s_bwd_pbrBSDF_0(&dp_kd_0, &dp_arm_0, &dp_pos_0, &dp_nrm_0, &dp_lint_0, _S275, _S286, min_roughness_6, _S282);



        float3  arm_accum_1 = arm_accum_0 + dp_arm_0.differential_0;
        float3  pos_accum_1 = pos_accum_0 + dp_pos_0.differential_0;
        float3  nrm_accum_1 = nrm_accum_0 + dp_nrm_0.differential_0;

#line 240
        kd_accum_0 = kd_accum_0 + dp_kd_0.differential_0;

#line 240
        arm_accum_0 = arm_accum_1;

#line 240
        pos_accum_0 = pos_accum_1;

#line 240
        nrm_accum_0 = nrm_accum_1;

#line 218
        i_5 = i_5 + 1U;

#line 218
    }

#line 246
    (kd_grad_0).store<float3 >((_S274), (_S269), (clamp_1(_slang_select(isfinite_0(kd_accum_0), kd_accum_0,_S277), _S284, _S285)));
    (arm_grad_0).store<float3 >((_S274), (_S269), (clamp_1(_slang_select(isfinite_0(arm_accum_0), arm_accum_0,_S277), _S284, _S285)));
    (pos_grad_0).store<float3 >((_S274), (_S269), (clamp_1(_slang_select(isfinite_0(pos_accum_0), pos_accum_0,_S277), _S284, _S285)));
    (nrm_grad_0).store<float3 >((_S274), (_S269), (clamp_1(_slang_select(isfinite_0(nrm_accum_0), nrm_accum_0,_S277), _S284, _S285)));
    return;
}

__global__ void pbr_nhwc_fwd_kernel(TensorView kd_3, TensorView arm_3, TensorView pos_3, TensorView nrm_5, TensorView view_pos_4, TensorView light_pos_4, TensorView light_intensity_3, float min_roughness_7, TensorView output_1)
{

#line 263
    uint3  _S288 = ((blockIdx)) * ((blockDim)) + ((threadIdx));
    uint _S289 = _S288.x;

#line 264
    uint _S290 = ((output_1).sizes[(2U)]);

#line 264
    bool _S291;

#line 264
    if(_S289 > _S290)
    {

#line 264
        _S291 = true;

#line 264
    }
    else
    {

#line 264
        uint _S292 = _S288.y;

#line 264
        uint _S293 = ((output_1).sizes[(1U)]);

#line 264
        _S291 = _S292 > _S293;

#line 264
    }

#line 264
    if(_S291)
    {

#line 264
        _S291 = true;

#line 264
    }
    else
    {

#line 264
        uint _S294 = _S288.z;

#line 264
        uint _S295 = ((output_1).sizes[(0U)]);

#line 264
        _S291 = _S294 > _S295;

#line 264
    }

#line 264
    if(_S291)
    {
        return;
    }

    uint _S296 = _S288.z;

#line 269
    uint _S297 = _S288.y;
    float3  _S298 = ((view_pos_4).load<float3>((_S296)));

    uint _S299 = ((light_pos_4).sizes[(1U)]);
    float3  _S300 = make_float3 (0.0f);

#line 273
    uint i_6 = 0U;

#line 273
    float3  res_2 = _S300;

    for(;;)
    {

#line 275
        if(i_6 < _S299)
        {
        }
        else
        {

#line 275
            break;
        }
        float3  _S301 = ((light_pos_4).load<float3>((_S296), (i_6)));
        float3  _S302 = ((light_intensity_3).load<float3>((_S296), (i_6)));
        if(!(length_0(_S302) > 0.0f))
        {
            i_6 = i_6 + 1U;

#line 275
            continue;
        }

#line 284
        float3  _S303 = ((kd_3).load<float3>((_S296), (_S297), (_S289)));

#line 284
        float3  _S304 = ((arm_3).load<float3>((_S296), (_S297), (_S289)));

#line 284
        float3  _S305 = ((pos_3).load<float3>((_S296), (_S297), (_S289)));

#line 284
        float3  _S306 = ((nrm_5).load<float3>((_S296), (_S297), (_S289)));

#line 284
        res_2 = res_2 + pbrBSDF_0(_S303, _S304, _S305, _S306, _S302, _S298, _S301, min_roughness_7);

#line 275
        i_6 = i_6 + 1U;

#line 275
    }

#line 286
    (output_1).store<float3 >((_S296), (_S297), (_S289), (res_2));
    return;
}

__global__ void pbr_nhwc_bwd_kernel(TensorView kd_4, TensorView arm_4, TensorView pos_4, TensorView nrm_6, TensorView view_pos_5, TensorView light_pos_5, TensorView light_intensity_4, float min_roughness_8, TensorView kd_grad_1, TensorView arm_grad_1, TensorView pos_grad_1, TensorView nrm_grad_1, TensorView light_intensity_grad_1, TensorView grad_out_1)
{

#line 305
    uint3  idx_1 = ((blockIdx)) * ((blockDim)) + ((threadIdx));
    uint _S307 = idx_1.x;

#line 306
    uint _S308 = ((grad_out_1).sizes[(2U)]);

#line 306
    bool _S309;

#line 306
    if(_S307 > _S308)
    {

#line 306
        _S309 = true;

#line 306
    }
    else
    {

#line 306
        uint _S310 = idx_1.y;

#line 306
        uint _S311 = ((grad_out_1).sizes[(1U)]);

#line 306
        _S309 = _S310 > _S311;

#line 306
    }

#line 306
    if(_S309)
    {

#line 306
        _S309 = true;

#line 306
    }
    else
    {

#line 306
        uint _S312 = idx_1.z;

#line 306
        uint _S313 = ((grad_out_1).sizes[(0U)]);

#line 306
        _S309 = _S312 > _S313;

#line 306
    }

#line 306
    if(_S309)
    {
        return;
    }

    uint _S314 = idx_1.z;

#line 311
    uint _S315 = idx_1.y;

    float3  _S316 = ((view_pos_5).load<float3>((_S314)));

    uint _S317 = ((light_pos_5).sizes[(1U)]);
    float3  _S318 = make_float3 (0.0f);

#line 321
    float3  _S319 = ((kd_4).load<float3>((_S314), (_S315), (_S307)));
    float3  _S320 = ((arm_4).load<float3>((_S314), (_S315), (_S307)));
    float3  _S321 = ((pos_4).load<float3>((_S314), (_S315), (_S307)));
    float3  _S322 = ((nrm_6).load<float3>((_S314), (_S315), (_S307)));
    float3  _S323 = ((grad_out_1).load<float3>((_S314), (_S315), (_S307)));

#line 325
    uint i_7 = 0U;

#line 325
    float3  kd_accum_1 = _S318;

#line 325
    float3  arm_accum_2 = _S318;

#line 325
    float3  pos_accum_2 = _S318;

#line 325
    float3  nrm_accum_2 = _S318;

#line 337
    float3  _S324 = make_float3 (0.0f);

#line 355
    float3  _S325 = make_float3 (-1.0f);

#line 355
    float3  _S326 = make_float3 (1.0f);

#line 327
    for(;;)
    {

#line 327
        if(i_7 < _S317)
        {
        }
        else
        {

#line 327
            break;
        }

        float3  _S327 = ((light_pos_5).load<float3>((_S314), (i_7)));
        float3  _S328 = ((light_intensity_4).load<float3>((_S314), (i_7)));
        if(!(length_0(_S328) > 0.0f))
        {
            i_7 = i_7 + 1U;

#line 327
            continue;
        }

#line 337
        DiffPair_vectorx3Cfloatx2C3x3E_0 dp_kd_1;

#line 337
        (&dp_kd_1)->primal_0 = _S319;

#line 337
        (&dp_kd_1)->differential_0 = _S324;
        DiffPair_vectorx3Cfloatx2C3x3E_0 dp_arm_1;

#line 338
        (&dp_arm_1)->primal_0 = _S320;

#line 338
        (&dp_arm_1)->differential_0 = _S324;
        DiffPair_vectorx3Cfloatx2C3x3E_0 dp_pos_1;

#line 339
        (&dp_pos_1)->primal_0 = _S321;

#line 339
        (&dp_pos_1)->differential_0 = _S324;
        DiffPair_vectorx3Cfloatx2C3x3E_0 dp_nrm_1;

#line 340
        (&dp_nrm_1)->primal_0 = _S322;

#line 340
        (&dp_nrm_1)->differential_0 = _S324;
        DiffPair_vectorx3Cfloatx2C3x3E_0 dp_lint_1;

#line 341
        (&dp_lint_1)->primal_0 = _S328;

#line 341
        (&dp_lint_1)->differential_0 = _S324;

        s_bwd_pbrBSDF_0(&dp_kd_1, &dp_arm_1, &dp_pos_1, &dp_nrm_1, &dp_lint_1, _S316, _S327, min_roughness_8, _S323);



        float3  arm_accum_3 = arm_accum_2 + dp_arm_1.differential_0;
        float3  pos_accum_3 = pos_accum_2 + dp_pos_1.differential_0;
        float3  nrm_accum_3 = nrm_accum_2 + dp_nrm_1.differential_0;

#line 349
        kd_accum_1 = kd_accum_1 + dp_kd_1.differential_0;

#line 349
        arm_accum_2 = arm_accum_3;

#line 349
        pos_accum_2 = pos_accum_3;

#line 349
        nrm_accum_2 = nrm_accum_3;

#line 327
        i_7 = i_7 + 1U;

#line 327
    }

#line 355
    (kd_grad_1).store<float3 >((_S314), (_S315), (_S307), (clamp_1(_slang_select(isfinite_0(kd_accum_1), kd_accum_1,_S318), _S325, _S326)));
    (arm_grad_1).store<float3 >((_S314), (_S315), (_S307), (clamp_1(_slang_select(isfinite_0(arm_accum_2), arm_accum_2,_S318), _S325, _S326)));
    (pos_grad_1).store<float3 >((_S314), (_S315), (_S307), (clamp_1(_slang_select(isfinite_0(pos_accum_2), pos_accum_2,_S318), _S325, _S326)));
    (nrm_grad_1).store<float3 >((_S314), (_S315), (_S307), (clamp_1(_slang_select(isfinite_0(nrm_accum_2), nrm_accum_2,_S318), _S325, _S326)));
    return;
}


#line 71
__device__ float3  lambertBSDF_0(float3  kd_5, float3  pos_5, float3  nrm_7, float3  light_intensity_5, float3  light_pos_6)
{

#line 84
    return kd_5 * make_float3 (lambert_0(nrm_7, normalize_0(light_pos_6 - pos_5))) * light_intensity_5;
}


#line 362
__global__ void lambert_bn_fwd_kernel(TensorView kd_6, TensorView pos_6, TensorView nrm_8, TensorView light_pos_7, TensorView light_intensity_6, TensorView output_2)
{

#line 369
    uint3  _S329 = ((blockIdx)) * ((blockDim)) + ((threadIdx));
    uint _S330 = _S329.x;

#line 370
    uint _S331 = ((output_2).sizes[(1U)]);

#line 370
    bool _S332;

#line 370
    if(_S330 > _S331)
    {

#line 370
        _S332 = true;

#line 370
    }
    else
    {

#line 370
        uint _S333 = _S329.y;

#line 370
        uint _S334 = ((output_2).sizes[(0U)]);

#line 370
        _S332 = _S333 > _S334;

#line 370
    }

#line 370
    if(_S332)
    {
        return;
    }

    uint _S335 = _S329.y;

    uint _S336 = ((light_pos_7).sizes[(1U)]);
    float3  _S337 = make_float3 (0.0f);

#line 378
    uint i_8 = 0U;

#line 378
    float3  res_3 = _S337;

    for(;;)
    {

#line 380
        if(i_8 < _S336)
        {
        }
        else
        {

#line 380
            break;
        }
        float3  _S338 = ((light_pos_7).load<float3>((_S335), (i_8)));
        float3  _S339 = ((light_intensity_6).load<float3>((_S335), (i_8)));
        if(!(length_0(_S339) > 0.0f))
        {
            i_8 = i_8 + 1U;

#line 380
            continue;
        }

#line 389
        float3  _S340 = ((kd_6).load<float3>((_S335), (_S330)));

#line 389
        float3  _S341 = ((pos_6).load<float3>((_S335), (_S330)));

#line 389
        float3  _S342 = ((nrm_8).load<float3>((_S335), (_S330)));

#line 389
        res_3 = res_3 + lambertBSDF_0(_S340, _S341, _S342, _S339, _S338);

#line 380
        i_8 = i_8 + 1U;

#line 380
    }

#line 392
    (output_2).store<float3 >((_S335), (_S330), (res_3));
    return;
}


#line 71
__device__ void s_bwd_prop_lambertBSDF_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpkd_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dppos_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpnrm_5, DiffPair_vectorx3Cfloatx2C3x3E_0 * dplight_intensity_1, float3  light_pos_8, float3  _s_dOut_9)
{

#line 77
    float3  _S343 = light_pos_8 - (*dppos_1).primal_0;

#line 77
    float3  _S344 = normalize_0(_S343);

#line 77
    float _S345 = s_primal_ctx_lambert_0((*dpnrm_5).primal_0, _S344);

#line 84
    float3  _S346 = (*dpkd_1).primal_0 * make_float3 (_S345) * _s_dOut_9;

#line 84
    float3  s_diff_diffuse_T_1 = (*dplight_intensity_1).primal_0 * _s_dOut_9;

#line 79
    float3  _S347 = (*dpkd_1).primal_0 * s_diff_diffuse_T_1;

#line 79
    float3  _S348 = make_float3 (_S345) * s_diff_diffuse_T_1;

#line 78
    float _S349 = _S347.x + _S347.y + _S347.z;

#line 78
    float3  _S350 = make_float3 (0.0f);

#line 78
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S351;

#line 78
    (&_S351)->primal_0 = (*dpnrm_5).primal_0;

#line 78
    (&_S351)->differential_0 = _S350;

#line 78
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S352;

#line 78
    (&_S352)->primal_0 = _S344;

#line 78
    (&_S352)->differential_0 = _S350;

#line 78
    s_bwd_prop_lambert_0(&_S351, &_S352, _S349);

#line 77
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S353;

#line 77
    (&_S353)->primal_0 = _S343;

#line 77
    (&_S353)->differential_0 = _S350;

#line 77
    s_bwd_normalize_impl_0(&_S353, _S352.differential_0);

#line 77
    float3  _S354 = - _S353.differential_0;

#line 77
    dplight_intensity_1->primal_0 = (*dplight_intensity_1).primal_0;

#line 77
    dplight_intensity_1->differential_0 = _S346;

#line 77
    dpnrm_5->primal_0 = (*dpnrm_5).primal_0;

#line 77
    dpnrm_5->differential_0 = _S351.differential_0;

#line 77
    dppos_1->primal_0 = (*dppos_1).primal_0;

#line 77
    dppos_1->differential_0 = _S354;

#line 77
    dpkd_1->primal_0 = (*dpkd_1).primal_0;

#line 77
    dpkd_1->differential_0 = _S348;

#line 71
    return;
}


#line 71
__device__ void s_bwd_lambertBSDF_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S355, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S356, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S357, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S358, float3  _S359, float3  _S360)
{


    s_bwd_prop_lambertBSDF_0(_S355, _S356, _S357, _S358, _S359, _S360);

#line 75
    return;
}


#line 396
__global__ void lambert_bn_bwd_kernel(TensorView kd_7, TensorView pos_7, TensorView nrm_9, TensorView light_pos_9, TensorView light_intensity_7, TensorView kd_grad_2, TensorView pos_grad_2, TensorView nrm_grad_2, TensorView lint_grad_0, TensorView grad_out_2)
{

#line 407
    uint3  idx_2 = ((blockIdx)) * ((blockDim)) + ((threadIdx));
    uint _S361 = idx_2.x;

#line 408
    uint _S362 = ((grad_out_2).sizes[(1U)]);

#line 408
    bool _S363;

#line 408
    if(_S361 > _S362)
    {

#line 408
        _S363 = true;

#line 408
    }
    else
    {

#line 408
        uint _S364 = idx_2.y;

#line 408
        uint _S365 = ((grad_out_2).sizes[(0U)]);

#line 408
        _S363 = _S364 > _S365;

#line 408
    }

#line 408
    if(_S363)
    {
        return;
    }

    uint _S366 = idx_2.y;

    uint _S367 = ((light_pos_9).sizes[(1U)]);
    float3  _S368 = make_float3 (0.0f);



    float3  _S369 = ((kd_7).load<float3>((_S366), (_S361)));
    float3  _S370 = ((pos_7).load<float3>((_S366), (_S361)));
    float3  _S371 = ((nrm_9).load<float3>((_S366), (_S361)));
    float3  _S372 = ((grad_out_2).load<float3>((_S366), (_S361)));

#line 423
    uint i_9 = 0U;

#line 423
    float3  kd_accum_2 = _S368;

#line 423
    float3  pos_accum_4 = _S368;

#line 423
    float3  nrm_accum_4 = _S368;

#line 435
    float3  _S373 = make_float3 (0.0f);

#line 450
    float3  _S374 = make_float3 (-1.0f);

#line 450
    float3  _S375 = make_float3 (1.0f);

#line 425
    for(;;)
    {

#line 425
        if(i_9 < _S367)
        {
        }
        else
        {

#line 425
            break;
        }

        float3  _S376 = ((light_pos_9).load<float3>((_S366), (i_9)));
        float3  _S377 = ((light_intensity_7).load<float3>((_S366), (i_9)));
        if(!(length_0(_S377) > 0.0f))
        {
            i_9 = i_9 + 1U;

#line 425
            continue;
        }

#line 435
        DiffPair_vectorx3Cfloatx2C3x3E_0 dp_kd_2;

#line 435
        (&dp_kd_2)->primal_0 = _S369;

#line 435
        (&dp_kd_2)->differential_0 = _S373;
        DiffPair_vectorx3Cfloatx2C3x3E_0 dp_pos_2;

#line 436
        (&dp_pos_2)->primal_0 = _S370;

#line 436
        (&dp_pos_2)->differential_0 = _S373;
        DiffPair_vectorx3Cfloatx2C3x3E_0 dp_nrm_2;

#line 437
        (&dp_nrm_2)->primal_0 = _S371;

#line 437
        (&dp_nrm_2)->differential_0 = _S373;
        DiffPair_vectorx3Cfloatx2C3x3E_0 dp_lint_2;

#line 438
        (&dp_lint_2)->primal_0 = _S377;

#line 438
        (&dp_lint_2)->differential_0 = _S373;

        s_bwd_lambertBSDF_0(&dp_kd_2, &dp_pos_2, &dp_nrm_2, &dp_lint_2, _S376, _S372);


        float3  pos_accum_5 = pos_accum_4 + dp_pos_2.differential_0;
        float3  nrm_accum_5 = nrm_accum_4 + dp_nrm_2.differential_0;

#line 444
        kd_accum_2 = kd_accum_2 + dp_kd_2.differential_0;

#line 444
        pos_accum_4 = pos_accum_5;

#line 444
        nrm_accum_4 = nrm_accum_5;

#line 425
        i_9 = i_9 + 1U;

#line 425
    }

#line 450
    (kd_grad_2).store<float3 >((_S366), (_S361), (clamp_1(_slang_select(isfinite_0(kd_accum_2), kd_accum_2,_S368), _S374, _S375)));
    (pos_grad_2).store<float3 >((_S366), (_S361), (clamp_1(_slang_select(isfinite_0(pos_accum_4), pos_accum_4,_S368), _S374, _S375)));
    (nrm_grad_2).store<float3 >((_S366), (_S361), (clamp_1(_slang_select(isfinite_0(nrm_accum_4), nrm_accum_4,_S368), _S374, _S375)));
    return;
}

__global__ void lambert_nhwc_fwd_kernel(TensorView kd_8, TensorView pos_8, TensorView nrm_10, TensorView light_pos_10, TensorView light_intensity_8, TensorView output_3)
{

#line 463
    uint3  _S378 = ((blockIdx)) * ((blockDim)) + ((threadIdx));
    uint _S379 = _S378.x;

#line 464
    uint _S380 = ((output_3).sizes[(2U)]);

#line 464
    bool _S381;

#line 464
    if(_S379 > _S380)
    {

#line 464
        _S381 = true;

#line 464
    }
    else
    {

#line 464
        uint _S382 = _S378.y;

#line 464
        uint _S383 = ((output_3).sizes[(1U)]);

#line 464
        _S381 = _S382 > _S383;

#line 464
    }

#line 464
    if(_S381)
    {

#line 464
        _S381 = true;

#line 464
    }
    else
    {

#line 464
        uint _S384 = _S378.z;

#line 464
        uint _S385 = ((output_3).sizes[(0U)]);

#line 464
        _S381 = _S384 > _S385;

#line 464
    }

#line 464
    if(_S381)
    {
        return;
    }

    uint _S386 = _S378.z;

#line 469
    uint _S387 = _S378.y;

    uint _S388 = ((light_pos_10).sizes[(1U)]);
    float3  _S389 = make_float3 (0.0f);

#line 472
    uint i_10 = 0U;

#line 472
    float3  res_4 = _S389;

    for(;;)
    {

#line 474
        if(i_10 < _S388)
        {
        }
        else
        {

#line 474
            break;
        }
        float3  _S390 = ((light_pos_10).load<float3>((_S386), (i_10)));
        float3  _S391 = ((light_intensity_8).load<float3>((_S386), (i_10)));

        float3  _S392 = ((kd_8).load<float3>((_S386), (_S387), (_S379)));

#line 479
        float3  _S393 = ((pos_8).load<float3>((_S386), (_S387), (_S379)));

#line 479
        float3  _S394 = ((nrm_10).load<float3>((_S386), (_S387), (_S379)));

#line 479
        float3  res_5 = res_4 + lambertBSDF_0(_S392, _S393, _S394, _S391, _S390);

#line 474
        i_10 = i_10 + 1U;

#line 474
        res_4 = res_5;

#line 474
    }

#line 482
    (output_3).store<float3 >((_S386), (_S387), (_S379), (res_4));
    return;
}

__global__ void lambert_nhwc_bwd_kernel(TensorView kd_9, TensorView pos_9, TensorView nrm_11, TensorView light_pos_11, TensorView light_intensity_9, TensorView kd_grad_3, TensorView pos_grad_3, TensorView nrm_grad_3, TensorView lint_grad_1, TensorView grad_out_3)
{

#line 497
    uint3  idx_3 = ((blockIdx)) * ((blockDim)) + ((threadIdx));
    uint _S395 = idx_3.x;

#line 498
    uint _S396 = ((grad_out_3).sizes[(2U)]);

#line 498
    bool _S397;

#line 498
    if(_S395 > _S396)
    {

#line 498
        _S397 = true;

#line 498
    }
    else
    {

#line 498
        uint _S398 = idx_3.y;

#line 498
        uint _S399 = ((grad_out_3).sizes[(1U)]);

#line 498
        _S397 = _S398 > _S399;

#line 498
    }

#line 498
    if(_S397)
    {

#line 498
        _S397 = true;

#line 498
    }
    else
    {

#line 498
        uint _S400 = idx_3.z;

#line 498
        uint _S401 = ((grad_out_3).sizes[(0U)]);

#line 498
        _S397 = _S400 > _S401;

#line 498
    }

#line 498
    if(_S397)
    {
        return;
    }

    uint _S402 = idx_3.z;

#line 503
    uint _S403 = idx_3.y;

    uint _S404 = ((light_pos_11).sizes[(1U)]);
    float3  _S405 = make_float3 (0.0f);



    float3  _S406 = ((kd_9).load<float3>((_S402), (_S403), (_S395)));
    float3  _S407 = ((pos_9).load<float3>((_S402), (_S403), (_S395)));
    float3  _S408 = ((nrm_11).load<float3>((_S402), (_S403), (_S395)));
    float3  _S409 = ((grad_out_3).load<float3>((_S402), (_S403), (_S395)));

#line 513
    uint i_11 = 0U;

#line 513
    float3  kd_accum_3 = _S405;

#line 513
    float3  pos_accum_6 = _S405;

#line 513
    float3  nrm_accum_6 = _S405;

#line 521
    float3  _S410 = make_float3 (0.0f);

#line 536
    float3  _S411 = make_float3 (-1.0f);

#line 536
    float3  _S412 = make_float3 (1.0f);

#line 515
    for(;;)
    {

#line 515
        if(i_11 < _S404)
        {
        }
        else
        {

#line 515
            break;
        }

        float3  _S413 = ((light_pos_11).load<float3>((_S402), (i_11)));
        float3  _S414 = ((light_intensity_9).load<float3>((_S402), (i_11)));

        DiffPair_vectorx3Cfloatx2C3x3E_0 dp_kd_3;

#line 521
        (&dp_kd_3)->primal_0 = _S406;

#line 521
        (&dp_kd_3)->differential_0 = _S410;
        DiffPair_vectorx3Cfloatx2C3x3E_0 dp_pos_3;

#line 522
        (&dp_pos_3)->primal_0 = _S407;

#line 522
        (&dp_pos_3)->differential_0 = _S410;
        DiffPair_vectorx3Cfloatx2C3x3E_0 dp_nrm_3;

#line 523
        (&dp_nrm_3)->primal_0 = _S408;

#line 523
        (&dp_nrm_3)->differential_0 = _S410;
        DiffPair_vectorx3Cfloatx2C3x3E_0 dp_lint_3;

#line 524
        (&dp_lint_3)->primal_0 = _S414;

#line 524
        (&dp_lint_3)->differential_0 = _S410;

        s_bwd_lambertBSDF_0(&dp_kd_3, &dp_pos_3, &dp_nrm_3, &dp_lint_3, _S413, _S409);

        float3  kd_accum_4 = kd_accum_3 + dp_kd_3.differential_0;
        float3  pos_accum_7 = pos_accum_6 + dp_pos_3.differential_0;
        float3  nrm_accum_7 = nrm_accum_6 + dp_nrm_3.differential_0;

#line 515
        i_11 = i_11 + 1U;

#line 515
        kd_accum_3 = kd_accum_4;

#line 515
        pos_accum_6 = pos_accum_7;

#line 515
        nrm_accum_6 = nrm_accum_7;

#line 515
    }

#line 536
    (kd_grad_3).store<float3 >((_S402), (_S403), (_S395), (clamp_1(_slang_select(isfinite_0(kd_accum_3), kd_accum_3,_S405), _S411, _S412)));
    (pos_grad_3).store<float3 >((_S402), (_S403), (_S395), (clamp_1(_slang_select(isfinite_0(pos_accum_6), pos_accum_6,_S405), _S411, _S412)));
    (nrm_grad_3).store<float3 >((_S402), (_S403), (_S395), (clamp_1(_slang_select(isfinite_0(nrm_accum_6), nrm_accum_6,_S405), _S411, _S412)));
    return;
}


#line 88
__device__ float3  specularBSDF_0(float3  kd_10, float3  arm_5, float3  pos_10, float3  nrm_12, float3  light_intensity_10, float3  view_pos_6, float3  light_pos_12, float min_roughness_9)
{

#line 98
    float3  _S415 = light_pos_12 - pos_10;

    float _S416 = arm_5.y;
    float _S417 = arm_5.z;
    float3  specular_0 = pbrSpecular_0((make_float3 (0.03999999910593033f * (1.0f - _S417)) + kd_10 * make_float3 (_S417)) * make_float3 (1.0f - arm_5.x), nrm_12, safeNormalize_0(view_pos_6 - pos_10), safeNormalize_0(_S415), _S416 * _S416, min_roughness_9);

    float _S418 = length_0(_S415);

#line 104
    float3  _S419;
    if(_S418 > 0.00100000004749745f)
    {

#line 105
        _S419 = light_intensity_10 / make_float3 (_S418);

#line 105
    }
    else
    {

#line 105
        _S419 = make_float3 (0.0f);

#line 105
    }
    return specular_0 * _S419;
}


#line 542
__global__ void specular_bn_fwd_kernel(TensorView kd_11, TensorView arm_6, TensorView pos_11, TensorView nrm_13, TensorView view_pos_7, TensorView light_pos_13, TensorView light_intensity_11, float min_roughness_10, TensorView output_4)
{

#line 552
    uint3  _S420 = ((blockIdx)) * ((blockDim)) + ((threadIdx));
    uint _S421 = _S420.x;

#line 553
    uint _S422 = ((output_4).sizes[(1U)]);

#line 553
    bool _S423;

#line 553
    if(_S421 > _S422)
    {

#line 553
        _S423 = true;

#line 553
    }
    else
    {

#line 553
        uint _S424 = _S420.y;

#line 553
        uint _S425 = ((output_4).sizes[(0U)]);

#line 553
        _S423 = _S424 > _S425;

#line 553
    }

#line 553
    if(_S423)
    {
        return;
    }

    uint _S426 = _S420.y;
    float3  _S427 = ((view_pos_7).load<float3>((_S426)));

    uint _S428 = ((light_pos_13).sizes[(1U)]);
    (output_4).store<float3 >((_S426), (_S421), (make_float3 (0.0f)));

#line 562
    uint i_12 = 0U;

    for(;;)
    {

#line 564
        if(i_12 < _S428)
        {
        }
        else
        {

#line 564
            break;
        }
        float3  _S429 = ((light_pos_13).load<float3>((_S426), (i_12)));
        float3  _S430 = ((light_intensity_11).load<float3>((_S426), (i_12)));
        if(!(length_0(_S430) > 0.0f))
        {
            i_12 = i_12 + 1U;

#line 564
            continue;
        }

#line 573
        float3  _S431 = ((kd_11).load<float3>((_S426), (_S421)));

#line 573
        float3  _S432 = ((arm_6).load<float3>((_S426), (_S421)));

#line 573
        float3  _S433 = ((pos_11).load<float3>((_S426), (_S421)));

#line 573
        float3  _S434 = ((nrm_13).load<float3>((_S426), (_S421)));

#line 573
        float3  _S435 = specularBSDF_0(_S431, _S432, _S433, _S434, _S430, _S427, _S429, min_roughness_10);
        float3  _S436 = ((output_4).load<float3>((_S426), (_S421)));

#line 574
        (output_4).store<float3 >((_S426), (_S421), (_S436 + _S435));

#line 564
        i_12 = i_12 + 1U;

#line 564
    }

#line 576
    return;
}


#line 88
__device__ void s_bwd_prop_specularBSDF_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpkd_2, DiffPair_vectorx3Cfloatx2C3x3E_0 * dparm_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dppos_2, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpnrm_6, DiffPair_vectorx3Cfloatx2C3x3E_0 * dplight_intensity_2, float3  view_pos_8, float3  light_pos_14, float min_roughness_11, float3  _s_dOut_10)
{

#line 95
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S437 = *dpkd_2;

#line 95
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S438 = *dpnrm_6;

#line 95
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S439 = *dplight_intensity_2;

#line 105
    float3  _S440 = make_float3 (0.0f);

#line 97
    float3  _S441 = view_pos_8 - (*dppos_2).primal_0;

#line 97
    float3  _S442 = s_primal_ctx_safeNormalize_0(_S441);
    float3  _S443 = light_pos_14 - (*dppos_2).primal_0;

#line 98
    float3  _S444 = s_primal_ctx_safeNormalize_0(_S443);

    float _S445 = (*dparm_1).primal_0.y;

#line 100
    float alpha_2 = _S445 * _S445;
    float _S446 = (*dparm_1).primal_0.z;

#line 101
    float3  _S447 = make_float3 (_S446);

#line 101
    float3  _S448 = make_float3 (0.03999999910593033f * (1.0f - _S446)) + (*dpkd_2).primal_0 * make_float3 (_S446);

#line 101
    float _S449 = 1.0f - (*dparm_1).primal_0.x;

#line 101
    float3  _S450 = make_float3 (_S449);

#line 101
    float3  spec_col_1 = _S448 * make_float3 (_S449);

#line 101
    float3  _S451 = s_primal_ctx_pbrSpecular_0(spec_col_1, (*dpnrm_6).primal_0, _S442, _S444, alpha_2, min_roughness_11);


    float _S452 = length_0(_S443);
    float3  _S453 = make_float3 (_S452);

#line 105
    bool _S454 = _S452 > 0.00100000004749745f;

#line 105
    float3  _S455;

#line 105
    float3  _S456;

#line 105
    if(_S454)
    {

#line 105
        float3  _S457 = make_float3 (_S452 * _S452);

#line 105
        _S455 = _S439.primal_0 / make_float3 (_S452);

#line 105
        _S456 = _S457;

#line 105
    }
    else
    {

#line 105
        _S455 = make_float3 (0.0f);

#line 105
        _S456 = _S440;

#line 105
    }
    float3  _S458 = _S451 * _s_dOut_10;

#line 106
    float3  _S459 = _S455 * _s_dOut_10;

#line 106
    if(_S454)
    {

#line 105
        float3  _S460 = _S458 / _S456;

#line 105
        float3  _S461 = _S453 * _S460;

#line 105
        _S455 = _S439.primal_0 * - _S460;

#line 105
        _S456 = _S461;

#line 105
    }
    else
    {

#line 105
        _S455 = _S440;

#line 105
        _S456 = _S440;

#line 105
    }

#line 104
    float _S462 = _S455.x + _S455.y + _S455.z;

#line 104
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S463;

#line 104
    (&_S463)->primal_0 = _S443;

#line 104
    (&_S463)->differential_0 = _S440;

#line 104
    s_bwd_length_impl_0(&_S463, _S462);

#line 102
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S464;

#line 102
    (&_S464)->primal_0 = spec_col_1;

#line 102
    (&_S464)->differential_0 = _S440;

#line 102
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S465;

#line 102
    (&_S465)->primal_0 = _S438.primal_0;

#line 102
    (&_S465)->differential_0 = _S440;

#line 102
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S466;

#line 102
    (&_S466)->primal_0 = _S442;

#line 102
    (&_S466)->differential_0 = _S440;

#line 102
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S467;

#line 102
    (&_S467)->primal_0 = _S444;

#line 102
    (&_S467)->differential_0 = _S440;

#line 102
    DiffPair_float_0 _S468;

#line 102
    (&_S468)->primal_0 = alpha_2;

#line 102
    (&_S468)->differential_0 = 0.0f;

#line 102
    s_bwd_prop_pbrSpecular_0(&_S464, &_S465, &_S466, &_S467, &_S468, min_roughness_11, _S459);

#line 101
    float3  _S469 = _S448 * _S464.differential_0;

#line 101
    float3  _S470 = _S450 * _S464.differential_0;

#line 101
    float _S471 = - (_S469.x + _S469.y + _S469.z);

#line 101
    float3  _S472 = _S437.primal_0 * _S470;

#line 101
    float3  _S473 = _S447 * _S470;

#line 101
    float _S474 = - (0.03999999910593033f * (_S470.x + _S470.y + _S470.z)) + _S472.x + _S472.y + _S472.z;

#line 100
    float _S475 = _S445 * _S468.differential_0;

#line 100
    float _S476 = _S475 + _S475;

#line 98
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S477;

#line 98
    (&_S477)->primal_0 = _S443;

#line 98
    (&_S477)->differential_0 = _S440;

#line 98
    s_bwd_prop_safeNormalize_0(&_S477, _S467.differential_0);

#line 98
    float3  _S478 = - (_S463.differential_0 + _S477.differential_0);

#line 97
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S479;

#line 97
    (&_S479)->primal_0 = _S441;

#line 97
    (&_S479)->differential_0 = _S440;

#line 97
    s_bwd_prop_safeNormalize_0(&_S479, _S466.differential_0);

#line 97
    float3  _S480 = - _S479.differential_0;

#line 97
    dplight_intensity_2->primal_0 = (*dplight_intensity_2).primal_0;

#line 97
    dplight_intensity_2->differential_0 = _S456;

#line 97
    dpnrm_6->primal_0 = (*dpnrm_6).primal_0;

#line 97
    dpnrm_6->differential_0 = _S465.differential_0;

#line 97
    float3  _S481 = _S478 + _S480;

#line 97
    dppos_2->primal_0 = (*dppos_2).primal_0;

#line 97
    dppos_2->differential_0 = _S481;

#line 97
    float3  _S482 = make_float3 (_S471, _S476, _S474);

#line 97
    dparm_1->primal_0 = (*dparm_1).primal_0;

#line 97
    dparm_1->differential_0 = _S482;

#line 97
    dpkd_2->primal_0 = (*dpkd_2).primal_0;

#line 97
    dpkd_2->differential_0 = _S473;

#line 88
    return;
}


#line 88
__device__ void s_bwd_specularBSDF_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S483, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S484, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S485, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S486, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S487, float3  _S488, float3  _S489, float _S490, float3  _S491)
{

#line 95
    s_bwd_prop_specularBSDF_0(_S483, _S484, _S485, _S486, _S487, _S488, _S489, _S490, _S491);

#line 95
    return;
}


#line 579
__global__ void specular_bn_bwd_kernel(TensorView kd_12, TensorView arm_7, TensorView pos_12, TensorView nrm_14, TensorView view_pos_9, TensorView light_pos_15, TensorView light_intensity_12, float min_roughness_12, TensorView kd_grad_4, TensorView arm_grad_2, TensorView pos_grad_4, TensorView nrm_grad_4, TensorView lint_grad_2, TensorView grad_out_4)
{

#line 594
    uint3  idx_4 = ((blockIdx)) * ((blockDim)) + ((threadIdx));
    uint _S492 = idx_4.x;

#line 595
    uint _S493 = ((grad_out_4).sizes[(1U)]);

#line 595
    bool _S494;

#line 595
    if(_S492 > _S493)
    {

#line 595
        _S494 = true;

#line 595
    }
    else
    {

#line 595
        uint _S495 = idx_4.y;

#line 595
        uint _S496 = ((grad_out_4).sizes[(0U)]);

#line 595
        _S494 = _S495 > _S496;

#line 595
    }

#line 595
    if(_S494)
    {
        return;
    }

    uint _S497 = idx_4.y;

    float3  _S498 = ((view_pos_9).load<float3>((_S497)));

    uint _S499 = ((light_pos_15).sizes[(1U)]);
    float3  _S500 = make_float3 (0.0f);

#line 610
    uint _S501 = ((kd_12).dimensionCount);

#line 610
    float lint_accum_0 = float(_S501);

    float3  _S502 = ((kd_12).load<float3>((_S497), (_S492)));
    float3  _S503 = ((arm_7).load<float3>((_S497), (_S492)));
    float3  _S504 = ((pos_12).load<float3>((_S497), (_S492)));
    float3  _S505 = ((nrm_14).load<float3>((_S497), (_S492)));
    float3  _S506 = ((grad_out_4).load<float3>((_S497), (_S492)));

#line 616
    uint i_13 = 0U;

#line 616
    float3  kd_accum_5 = _S500;

#line 616
    float3  arm_accum_4 = _S500;

#line 616
    float3  pos_accum_8 = _S500;

#line 616
    float3  nrm_accum_8 = _S500;

#line 628
    float3  _S507 = make_float3 (0.0f);

#line 645
    float3  _S508 = make_float3 (-1.0f);

#line 645
    float3  _S509 = make_float3 (1.0f);



    bool _S510 = (F32_isfinite((lint_accum_0)));

#line 618
    for(;;)
    {

#line 618
        if(i_13 < _S499)
        {
        }
        else
        {

#line 618
            break;
        }

        float3  _S511 = ((light_pos_15).load<float3>((_S497), (i_13)));
        float3  _S512 = ((light_intensity_12).load<float3>((_S497), (i_13)));
        if(!(length_0(_S512) > 0.0f))
        {
            i_13 = i_13 + 1U;

#line 618
            continue;
        }

#line 628
        DiffPair_vectorx3Cfloatx2C3x3E_0 dp_kd_4;

#line 628
        (&dp_kd_4)->primal_0 = _S502;

#line 628
        (&dp_kd_4)->differential_0 = _S507;
        DiffPair_vectorx3Cfloatx2C3x3E_0 dp_arm_2;

#line 629
        (&dp_arm_2)->primal_0 = _S503;

#line 629
        (&dp_arm_2)->differential_0 = _S507;
        DiffPair_vectorx3Cfloatx2C3x3E_0 dp_pos_4;

#line 630
        (&dp_pos_4)->primal_0 = _S504;

#line 630
        (&dp_pos_4)->differential_0 = _S507;
        DiffPair_vectorx3Cfloatx2C3x3E_0 dp_nrm_4;

#line 631
        (&dp_nrm_4)->primal_0 = _S505;

#line 631
        (&dp_nrm_4)->differential_0 = _S507;
        DiffPair_vectorx3Cfloatx2C3x3E_0 dp_lint_4;

#line 632
        (&dp_lint_4)->primal_0 = _S512;

#line 632
        (&dp_lint_4)->differential_0 = _S507;

        s_bwd_specularBSDF_0(&dp_kd_4, &dp_arm_2, &dp_pos_4, &dp_nrm_4, &dp_lint_4, _S498, _S511, min_roughness_12, _S506);


        float3  arm_accum_5 = arm_accum_4 + dp_arm_2.differential_0;
        float3  pos_accum_9 = pos_accum_8 + dp_pos_4.differential_0;
        float3  nrm_accum_9 = nrm_accum_8 + dp_nrm_4.differential_0;

#line 639
        kd_accum_5 = kd_accum_5 + dp_kd_4.differential_0;

#line 639
        arm_accum_4 = arm_accum_5;

#line 639
        pos_accum_8 = pos_accum_9;

#line 639
        nrm_accum_8 = nrm_accum_9;

#line 618
        i_13 = i_13 + 1U;

#line 618
    }

#line 645
    (kd_grad_4).store<float3 >((_S497), (_S492), (clamp_1(_slang_select(isfinite_0(kd_accum_5), kd_accum_5,_S500), _S508, _S509)));
    (arm_grad_2).store<float3 >((_S497), (_S492), (clamp_1(_slang_select(isfinite_0(arm_accum_4), arm_accum_4,_S500), _S508, _S509)));
    (pos_grad_4).store<float3 >((_S497), (_S492), (clamp_1(_slang_select(isfinite_0(pos_accum_8), pos_accum_8,_S500), _S508, _S509)));
    (nrm_grad_4).store<float3 >((_S497), (_S492), (clamp_1(_slang_select(isfinite_0(nrm_accum_8), nrm_accum_8,_S500), _S508, _S509)));
    (lint_grad_2).store<float3 >((_S497), (make_float3 (clamp_0(_slang_select(_S510, lint_accum_0,0.0f), -1.0f, 1.0f))));
    return;
}


/*===-- dpi/svdpi.h - SystemVerilog Direct Programming Interface --*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This file contains the constant definitions, structure definitions,        *|
|* and routine declarations used by SystemVerilog DPI.                        *|
|*                                                                            *|
|* This file is from the SystemVerilog IEEE 1800-2017 Annex I.                *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef EXTERNAL_DPI_SVDPI_H
#define EXTERNAL_DPI_SVDPI_H

#ifdef __cplusplus
extern "C" {
#endif

/* Define size-critical types on all OS platforms. */
#if defined(_MSC_VER)
typedef unsigned __int64 uint64_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int8 uint8_t;
typedef signed __int64 int64_t;
typedef signed __int32 int32_t;
typedef signed __int8 int8_t;
#elif defined(__MINGW32__)
#include <stdint.h>
#elif defined(__APPLE__)
#include <stdint.h>
#elif defined(__linux) || (defined(__APPLE__) && defined(__MACH__))
#include <inttypes.h>
#else
#include <sys/types.h>
#endif

/* Use to import a symbol into dll */
#ifndef DPI_DLLISPEC
#if (defined(_MSC_VER) || defined(__MINGW32__) || defined(__CYGWIN__))
#define DPI_DLLISPEC __declspec(dllimport)
#else
#define DPI_DLLISPEC
#endif
#endif

/* Use to export a symbol from dll */
#ifndef DPI_DLLESPEC
#if (defined(_MSC_VER) || defined(__MINGW32__) || defined(__CYGWIN__))
#define DPI_DLLESPEC __declspec(dllexport)
#else
#define DPI_DLLESPEC
#endif
#endif

/* Use to mark a function as external */
#ifndef DPI_EXTERN
#define DPI_EXTERN
#endif

#ifndef DPI_PROTOTYPES
#define DPI_PROTOTYPES
/* object is defined imported by the application */
#define XXTERN DPI_EXTERN DPI_DLLISPEC
/* object is exported by the application */
#define EETERN DPI_EXTERN DPI_DLLESPEC
#endif

/* canonical    representation */
#define sv_0 0
#define sv_1 1
#define sv_z 2
#define sv_x 3

/* common type for 'bit' and 'logic' scalars. */
typedef uint8_t svScalar;
typedef svScalar svBit;   /* scalar */
typedef svScalar svLogic; /* scalar */

/*
 * DPI representation of packed arrays.
 * 2-state and 4-state vectors, exactly the same as PLI's avalue/bvalue.
 */
#ifndef VPI_VECVAL
#define VPI_VECVAL
typedef struct TVpiVecval {
  uint32_t aval;
  uint32_t bval;
} s_vpi_vecval, *p_vpi_vecval;
#endif

/* (a chunk of) packed logic array */
typedef s_vpi_vecval svLogicVecVal;

/* (a chunk of) packed bit array */
typedef uint32_t svBitVecVal;

/* Number of chunks required to represent the given width packed array */
#define SV_PACKED_DATA_NELEMS(WIDTH) (((WIDTH) + 31) >> 5)

/*
 * Because the contents of the unused bits is undetermined,
 * the following macros can be handy.
 */
#define SV_MASK(N) (~(-1 << (N)))

#define SV_GET_UNSIGNED_BITS(VALUE, N)                                         \
  ((N) == 32 ? (VALUE) : ((VALUE)&SV_MASK(N)))

#define SV_GET_SIGNED_BITS(VALUE, N)                                           \
  ((N) == 32 ? (VALUE)                                                         \
             : (((VALUE) & (1 << (N))) ? ((VALUE) | ~SV_MASK(N))               \
                                       : ((VALUE)&SV_MASK(N))))

/*
 * Implementation-dependent representation.
 */
/*
 * Return implementation version information string ("1800-2005" or "SV3.1a").
 */
XXTERN const char *svDpiVersion(void);

/* a handle to a scope (an instance of a module or interface) */
XXTERN typedef void *svScope;

/* a handle to a generic object (actually, unsized array) */
XXTERN typedef void *svOpenArrayHandle;

/*
 * Bit-select utility functions.
 *
 * Packed arrays are assumed to be indexed n-1:0,
 * where 0 is the index of LSB
 */

/* s=source, i=bit-index */
XXTERN svBit svGetBitselBit(const svBitVecVal *s, int i);
XXTERN svLogic svGetBitselLogic(const svLogicVecVal *s, int i);

/* d=destination, i=bit-index, s=scalar */
XXTERN void svPutBitselBit(svBitVecVal *d, int i, svBit s);
XXTERN void svPutBitselLogic(svLogicVecVal *d, int i, svLogic s);

/*
 * Part-select utility functions.
 *
 * A narrow (<=32 bits) part-select is extracted from the
 * source representation and written into the destination word.
 *
 * Normalized ranges and indexing [n-1:0] are used for both arrays.
 *
 * s=source, d=destination, i=starting bit index, w=width
 * like for variable part-selects; limitations: w <= 32
 */
XXTERN void svGetPartselBit(svBitVecVal *d, const svBitVecVal *s, int i, int w);
XXTERN void svGetPartselLogic(svLogicVecVal *d, const svLogicVecVal *s, int i,
                              int w);

XXTERN void svPutPartselBit(svBitVecVal *d, const svBitVecVal s, int i, int w);
XXTERN void svPutPartselLogic(svLogicVecVal *d, const svLogicVecVal s, int i,
                              int w);

/*
 * Open array querying functions
 * These functions are modeled upon the SystemVerilog array
 * querying functions and use the same semantics.
 *
 * If the dimension is 0, then the query refers to the
 * packed part of an array (which is one-dimensional).
 * Dimensions > 0 refer to the unpacked part of an array.
 */
/* h= handle to open array, d=dimension */
XXTERN int svLeft(const svOpenArrayHandle h, int d);
XXTERN int svRight(const svOpenArrayHandle h, int d);
XXTERN int svLow(const svOpenArrayHandle h, int d);
XXTERN int svHigh(const svOpenArrayHandle h, int d);
XXTERN int svIncrement(const svOpenArrayHandle h, int d);
XXTERN int svSize(const svOpenArrayHandle h, int d);
XXTERN int svDimensions(const svOpenArrayHandle h);

/*
 * Pointer to the actual representation of the whole array of any type
 * NULL if not in C layout
 */
XXTERN void *svGetArrayPtr(const svOpenArrayHandle);

/* total size in bytes or 0 if not in C layout */
XXTERN int svSizeOfArray(const svOpenArrayHandle);

/*
 * Return a pointer to an element of the array
 * or NULL if index outside the range or null pointer
 */
XXTERN void *svGetArrElemPtr(const svOpenArrayHandle, int indx1, ...);

/* specialized versions for 1-, 2- and 3-dimensional arrays: */
XXTERN void *svGetArrElemPtr1(const svOpenArrayHandle, int indx1);
XXTERN void *svGetArrElemPtr2(const svOpenArrayHandle, int indx1, int indx2);
XXTERN void *svGetArrElemPtr3(const svOpenArrayHandle, int indx1, int indx2,
                              int indx3);

/*
 * Functions for copying between simulator storage and user space.
 * These functions copy the whole packed array in either direction.
 * The user is responsible for allocating an array to hold the
 * canonical representation.
 */

/* s=source, d=destination */
/* From user space into simulator storage */
XXTERN void svPutBitArrElemVecVal(const svOpenArrayHandle d,
                                  const svBitVecVal *s, int indx1, ...);
XXTERN void svPutBitArrElem1VecVal(const svOpenArrayHandle d,
                                   const svBitVecVal *s, int indx1);
XXTERN void svPutBitArrElem2VecVal(const svOpenArrayHandle d,
                                   const svBitVecVal *s, int indx1, int indx2);
XXTERN void svPutBitArrElem3VecVal(const svOpenArrayHandle d,
                                   const svBitVecVal *s, int indx1, int indx2,
                                   int indx3);

XXTERN void svPutLogicArrElemVecVal(const svOpenArrayHandle d,
                                    const svLogicVecVal *s, int indx1, ...);
XXTERN void svPutLogicArrElem1VecVal(const svOpenArrayHandle d,
                                     const svLogicVecVal *s, int indx1);
XXTERN void svPutLogicArrElem2VecVal(const svOpenArrayHandle d,
                                     const svLogicVecVal *s, int indx1,
                                     int indx2);
XXTERN void svPutLogicArrElem3VecVal(const svOpenArrayHandle d,
                                     const svLogicVecVal *s, int indx1,
                                     int indx2, int indx3);

/* From simulator storage into user space */
XXTERN void svGetBitArrElemVecVal(svBitVecVal *d, const svOpenArrayHandle s,
                                  int indx1, ...);
XXTERN void svGetBitArrElem1VecVal(svBitVecVal *d, const svOpenArrayHandle s,
                                   int indx1);
XXTERN void svGetBitArrElem2VecVal(svBitVecVal *d, const svOpenArrayHandle s,
                                   int indx1, int indx2);
XXTERN void svGetBitArrElem3VecVal(svBitVecVal *d, const svOpenArrayHandle s,
                                   int indx1, int indx2, int indx3);
XXTERN void svGetLogicArrElemVecVal(svLogicVecVal *d, const svOpenArrayHandle s,
                                    int indx1, ...);
XXTERN void svGetLogicArrElem1VecVal(svLogicVecVal *d,
                                     const svOpenArrayHandle s, int indx1);
XXTERN void svGetLogicArrElem2VecVal(svLogicVecVal *d,
                                     const svOpenArrayHandle s, int indx1,
                                     int indx2);
XXTERN void svGetLogicArrElem3VecVal(svLogicVecVal *d,
                                     const svOpenArrayHandle s, int indx1,
                                     int indx2, int indx3);

XXTERN svBit svGetBitArrElem(const svOpenArrayHandle s, int indx1, ...);
XXTERN svBit svGetBitArrElem1(const svOpenArrayHandle s, int indx1);
XXTERN svBit svGetBitArrElem2(const svOpenArrayHandle s, int indx1, int indx2);
XXTERN svBit svGetBitArrElem3(const svOpenArrayHandle s, int indx1, int indx2,
                              int indx3);
XXTERN svLogic svGetLogicArrElem(const svOpenArrayHandle s, int indx1, ...);
XXTERN svLogic svGetLogicArrElem1(const svOpenArrayHandle s, int indx1);
XXTERN svLogic svGetLogicArrElem2(const svOpenArrayHandle s, int indx1,
                                  int indx2);
XXTERN svLogic svGetLogicArrElem3(const svOpenArrayHandle s, int indx1,
                                  int indx2, int indx3);
XXTERN void svPutLogicArrElem(const svOpenArrayHandle d, svLogic value,
                              int indx1, ...);
XXTERN void svPutLogicArrElem1(const svOpenArrayHandle d, svLogic value,
                               int indx1);
XXTERN void svPutLogicArrElem2(const svOpenArrayHandle d, svLogic value,
                               int indx1, int indx2);
XXTERN void svPutLogicArrElem3(const svOpenArrayHandle d, svLogic value,
                               int indx1, int indx2, int indx3);
XXTERN void svPutBitArrElem(const svOpenArrayHandle d, svBit value, int indx1,
                            ...);
XXTERN void svPutBitArrElem1(const svOpenArrayHandle d, svBit value, int indx1);
XXTERN void svPutBitArrElem2(const svOpenArrayHandle d, svBit value, int indx1,
                             int indx2);
XXTERN void svPutBitArrElem3(const svOpenArrayHandle d, svBit value, int indx1,
                             int indx2, int indx3);

/* Functions for working with DPI context */

/*
 * Retrieve the active instance scope currently associated with the executing
 * imported function. Unless a prior call to svSetScope has occurred, this
 * is the scope of the function's declaration site, not call site.
 * Returns NULL if called from C code that is *not* an imported function.
 */
XXTERN svScope svGetScope(void);

/*
 * Set context for subsequent export function execution.
 * This function must be called before calling an export function, unless
 * the export function is called while executing an import function. In that
 * case the export function shall inherit the scope of the surrounding import
 * function. This is known as the "default scope".
 * The return is the previous active scope (per svGetScope)
 */
XXTERN svScope svSetScope(const svScope scope);

/* Gets the fully qualified name of a scope handle */
XXTERN const char *svGetNameFromScope(const svScope);

/*
 * Retrieve svScope to instance scope of an arbitrary function declaration.
 * (can be either module, program, interface, or generate scope)
 * The return value shall be NULL for unrecognized scope names.
 */
XXTERN svScope svGetScopeFromName(const char *scopeName);

/*
 * Store an arbitrary user data pointer for later retrieval by svGetUserData()
 * The userKey is generated by the user. It must be guaranteed by the user to
 * be unique from all other userKey's for all unique data storage requirements
 * It is recommended that the address of static functions or variables in the
 * user's C code be used as the userKey.
 * It is illegal to pass in NULL values for either the scope or userData
 * arguments. It is also an error to call svPutUserData() with an invalid
 * svScope. This function returns -1 for all error cases, 0 upon success. It is
 * suggested that userData values of 0 (NULL) not be used as otherwise it can
 * be impossible to discern error status returns when calling svGetUserData()
 */
XXTERN int svPutUserData(const svScope scope, void *userKey, void *userData);

/*
 * Retrieve an arbitrary user data pointer that was previously
 * stored by a call to svPutUserData(). See the comment above
 * svPutUserData() for an explanation of userKey, as well as
 * restrictions on NULL and illegal svScope and userKey values.
 * This function returns NULL for all error cases, 0 upon success.
 * This function also returns NULL in the event that a prior call
 * to svPutUserData() was never made.
 */
XXTERN void *svGetUserData(const svScope scope, void *userKey);

/*
 * Returns the file and line number in the SV code from which the import call
 * was made. If this information available, returns TRUE and updates fileName
 * and lineNumber to the appropriate values. Behavior is unpredictable if
 * fileName or lineNumber are not appropriate pointers. If this information is
 * not available return FALSE and contents of fileName and lineNumber not
 * modified. Whether this information is available or not is implementation-
 * specific. Note that the string provided (if any) is owned by the SV
 * implementation and is valid only until the next call to any SV function.
 * Applications must not modify this string or free it
 */
XXTERN int svGetCallerInfo(const char **fileName, int *lineNumber);

/*
 * Returns 1 if the current execution thread is in the disabled state.
 * Disable protocol must be adhered to if in the disabled state.
 */
XXTERN int svIsDisabledState(void);

/*
 * Imported functions call this API function during disable processing to
 * acknowledge that they are correctly participating in the DPI disable
 * protocol. This function must be called before returning from an imported
 * function that is in the disabled state.
 */
XXTERN void svAckDisabledState(void);

/*
 **********************************************************
 * DEPRECATED PORTION OF FILE ENDS REMOVED.
 * So that we don't accidently use them
 **********************************************************
 */

#undef DPI_EXTERN

#ifdef DPI_PROTOTYPES
#undef DPI_PROTOTYPES
#undef XXTERN
#undef EETERN
#endif

#ifdef __cplusplus
}
#endif

#endif

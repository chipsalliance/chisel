//===-- Moore.h - C API for Moore dialect ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_MOORE_H
#define CIRCT_C_DIALECT_MOORE_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Moore, moore);

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

enum MooreIntKind {
  // The integer vector types. These are the builtin single-bit integer types.
  /// A `bit`.
  MooreBit,
  /// A `logic`.
  MooreLogic,
  /// A `reg`.
  MooreReg,

  // The integer atom types. These are the builtin multi-bit integer types.
  /// A `byte`.
  MooreByte,
  /// A `shortint`.
  MooreShortInt,
  /// An `int`.
  MooreInt,
  /// A `longint`.
  MooreLongInt,
  /// An `integer`.
  MooreInteger,
  /// A `time`.
  MooreTime,
};

enum MooreRealKind {
  /// A `shortreal`.
  MooreShortReal,
  /// A `real`.
  MooreReal,
  /// A `realtime`.
  MooreRealTime,
};

enum MooreSign {
  /// No sign is explicitly given.
  MooreNone,
  /// Explicitly marked to be unsigned.
  MooreUnsigned,
  /// Explicitly marked to be signed.
  MooreSigned,
};

/// Create a void type.
MLIR_CAPI_EXPORTED MlirType mooreVoidTypeGet(MlirContext ctx);
/// Create a string type.
MLIR_CAPI_EXPORTED MlirType mooreStringTypeGet(MlirContext ctx);
/// Create a chandle type.
MLIR_CAPI_EXPORTED MlirType mooreChandleTypeGet(MlirContext ctx);
/// Create an event type.
MLIR_CAPI_EXPORTED MlirType mooreEventTypeGet(MlirContext ctx);
/// Create an int type.
MLIR_CAPI_EXPORTED MlirType mooreIntTypeGet(MlirContext ctx,
                                            enum MooreIntKind kind,
                                            enum MooreSign sign);
/// Create a `logic` type.
MLIR_CAPI_EXPORTED MlirType mooreIntTypeGetLogic(MlirContext ctx);
/// Create an `int` type.
MLIR_CAPI_EXPORTED MlirType mooreIntTypeGetInt(MlirContext ctx);
/// Create a `time` type.
MLIR_CAPI_EXPORTED MlirType mooreIntTypeGetTime(MlirContext ctx);
/// Create a real type.
MLIR_CAPI_EXPORTED MlirType mooreRealTypeGet(MlirContext ctx,
                                             enum MooreRealKind kind);
/// Create a packed unsized dimension type.
MLIR_CAPI_EXPORTED MlirType moorePackedUnsizedDimTypeGet(MlirType inner);
/// Create a packed range dimension type.
MLIR_CAPI_EXPORTED MlirType moorePackedRangeDimTypeGet(MlirType inner,
                                                       unsigned size,
                                                       bool upDir, int offset);
/// Create a unpacked unsized dimension type.
MLIR_CAPI_EXPORTED MlirType mooreUnpackedUnsizedDimTypeGet(MlirType inner);
/// Create a unpacked array dimension type.
MLIR_CAPI_EXPORTED MlirType mooreUnpackedArrayDimTypeGet(MlirType inner,
                                                         unsigned size);
/// Create a unpacked range dimension type.
MLIR_CAPI_EXPORTED MlirType mooreUnpackedRangeDimTypeGet(MlirType inner,
                                                         unsigned size,
                                                         bool upDir,
                                                         int offset);
/// Create a unpacked assoc dimension type without index.
MLIR_CAPI_EXPORTED MlirType mooreUnpackedAssocDimTypeGet(MlirType inner);
/// Create a unpacked assoc dimension type width index.
MLIR_CAPI_EXPORTED MlirType
mooreUnpackedAssocDimTypeGetWithIndex(MlirType inner, MlirType indexType);
/// Create a unpacked queue dimension type without bound.
MLIR_CAPI_EXPORTED MlirType mooreUnpackedQueueDimTypeGet(MlirType inner);
/// Create a unpacked queue dimension type with bound.
MLIR_CAPI_EXPORTED MlirType
mooreUnpackedQueueDimTypeGetWithBound(MlirType inner, unsigned bound);
/// Create a enum type without base.
MLIR_CAPI_EXPORTED MlirType mooreEnumTypeGet(MlirAttribute name,
                                             MlirLocation loc);
/// Create a enum type with base.
MLIR_CAPI_EXPORTED MlirType mooreEnumTypeGetWithBase(MlirAttribute name,
                                                     MlirLocation loc,
                                                     MlirType base);
// TODO: PackedStructType
// TODO: UnpackedStructType
/// Create a simple bit-vector type.
MLIR_CAPI_EXPORTED MlirType mooreSimpleBitVectorTypeGet(MlirContext ctx,
                                                        bool isFourValued,
                                                        bool isSigned,
                                                        unsigned size);
/// Checks whether the passed UnpackedType is a four-valued type.
MLIR_CAPI_EXPORTED bool mooreIsFourValuedType(MlirType type);
/// Checks whether the passed type is a simple bit-vector.
MLIR_CAPI_EXPORTED bool mooreIsSimpleBitVectorType(MlirType type);
/// Returns the size of a simple bit-vector type in bits.
MLIR_CAPI_EXPORTED unsigned mooreGetSimpleBitVectorSize(MlirType type);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_MOORE_H

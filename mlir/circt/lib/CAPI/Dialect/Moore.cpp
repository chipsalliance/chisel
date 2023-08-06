//===-- Moore.h - C API for Moore dialect ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/Moore.h"
#include "circt/Dialect/Moore/MooreDialect.h"
#include "circt/Dialect/Moore/MooreTypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

using namespace circt;
using namespace circt::moore;

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Moore, moore, MooreDialect)

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

static std::optional<Sign> convertMooreSign(enum MooreSign sign) {
  switch (sign) {
  case MooreSign::MooreSigned:
    return Sign::Signed;
  case MooreSign::MooreUnsigned:
    return Sign::Unsigned;
  case MooreSign::MooreNone:
    return {};
  }
  llvm_unreachable("All cases should be covered.");
}

static IntType::Kind convertMooreIntKind(enum MooreIntKind kind) {
  switch (kind) {
  case MooreIntKind::MooreBit:
    return IntType::Kind::Bit;
  case MooreIntKind::MooreLogic:
    return IntType::Kind::Logic;
  case MooreIntKind::MooreReg:
    return IntType::Kind::Reg;

  case MooreIntKind::MooreByte:
    return IntType::Kind::Byte;
  case MooreIntKind::MooreShortInt:
    return IntType::Kind::ShortInt;
  case MooreIntKind::MooreInt:
    return IntType::Kind::Int;
  case MooreIntKind::MooreLongInt:
    return IntType::Kind::LongInt;
  case MooreIntKind::MooreInteger:
    return IntType::Kind::Integer;
  case MooreIntKind::MooreTime:
    return IntType::Kind::Time;
  }
  llvm_unreachable("All cases should be covered.");
}

static RealType::Kind convertMooreRealKind(enum MooreRealKind kind) {
  switch (kind) {
  case MooreRealKind::MooreShortReal:
    return circt::moore::RealType::ShortReal;
  case MooreRealKind::MooreReal:
    return circt::moore::RealType::Real;
  case MooreRealKind::MooreRealTime:
    return circt::moore::RealType::RealTime;
  }
  llvm_unreachable("All cases should be covered.");
}

/// Create a void type.
MlirType mooreVoidTypeGet(MlirContext ctx) {
  return wrap(VoidType::get(unwrap(ctx)));
}

/// Create a string type.
MlirType mooreStringTypeGet(MlirContext ctx) {
  return wrap(StringType::get(unwrap(ctx)));
}

/// Create a chandle type.
MlirType mooreChandleTypeGet(MlirContext ctx) {
  return wrap(ChandleType::get(unwrap(ctx)));
}

/// Create a event type.
MlirType mooreEventTypeGet(MlirContext ctx) {
  return wrap(EventType::get(unwrap(ctx)));
}

/// Create an int type.
MlirType mooreIntTypeGet(MlirContext ctx, enum MooreIntKind kind,
                         enum MooreSign sign) {
  return wrap(IntType::get(unwrap(ctx), convertMooreIntKind(kind),
                           convertMooreSign(sign)));
}

/// Create a `logic` type.
MlirType mooreIntTypeGetLogic(MlirContext ctx) {
  return wrap(IntType::getLogic(unwrap(ctx)));
}

/// Create an `int` type.
MlirType mooreIntTypeGetInt(MlirContext ctx) {
  return wrap(IntType::getInt(unwrap(ctx)));
}

/// Create a `time` type.
MlirType mooreIntTypeGetTime(MlirContext ctx) {
  return wrap(IntType::getTime(unwrap(ctx)));
}

/// Create a real type.
MlirType mooreRealTypeGet(MlirContext ctx, enum MooreRealKind kind) {
  return wrap(RealType::get(unwrap(ctx), convertMooreRealKind(kind)));
}

/// Create a packed unsized dimension type.
MlirType moorePackedUnsizedDimTypeGet(MlirType inner) {
  return wrap(PackedUnsizedDim::get(unwrap(inner).cast<PackedType>()));
}

/// Create a packed range dimension type.
MlirType moorePackedRangeDimTypeGet(MlirType inner, unsigned size, bool upDir,
                                    int offset) {
  RangeDir dir = upDir ? RangeDir::Up : RangeDir::Down;
  return wrap(
      PackedRangeDim::get(unwrap(inner).cast<PackedType>(), size, dir, offset));
}

/// Create a unpacked unsized dimension type.
MlirType mooreUnpackedUnsizedDimTypeGet(MlirType inner) {
  return wrap(UnpackedUnsizedDim::get(unwrap(inner).cast<UnpackedType>()));
}

/// Create a unpacked array dimension type.
MlirType mooreUnpackedArrayDimTypeGet(MlirType inner, unsigned size) {
  return wrap(UnpackedArrayDim::get(unwrap(inner).cast<UnpackedType>(), size));
}

/// Create a unpacked range dimension type.
MlirType mooreUnpackedRangeDimTypeGet(MlirType inner, unsigned size, bool upDir,
                                      int offset) {
  RangeDir dir = upDir ? RangeDir::Up : RangeDir::Down;
  return wrap(UnpackedRangeDim::get(unwrap(inner).cast<UnpackedType>(), size,
                                    dir, offset));
}

/// Create a unpacked assoc dimension type without index.
MlirType mooreUnpackedAssocDimTypeGet(MlirType inner) {
  return wrap(UnpackedAssocDim::get(unwrap(inner).cast<UnpackedType>()));
}

/// Create a unpacked assoc dimension type with index.
MlirType mooreUnpackedAssocDimTypeGetWithIndex(MlirType inner,
                                               MlirType indexType) {
  return wrap(UnpackedAssocDim::get(unwrap(inner).cast<UnpackedType>(),
                                    unwrap(indexType).cast<UnpackedType>()));
}

/// Create a unpacked queue dimension type without bound.
MlirType mooreUnpackedQueueDimTypeGet(MlirType inner) {
  return wrap(UnpackedQueueDim::get(unwrap(inner).cast<UnpackedType>()));
}

/// Create a unpacked queue dimension type with bound.
MlirType mooreUnpackedQueueDimTypeGetWithBound(MlirType inner, unsigned bound) {
  return wrap(UnpackedQueueDim::get(unwrap(inner).cast<UnpackedType>(), bound));
}

/// Create a enum type without base.
MlirType mooreEnumTypeGet(MlirAttribute name, MlirLocation loc) {
  return wrap(EnumType::get(unwrap(name).cast<StringAttr>(), unwrap(loc)));
}

/// Create a enum type width base.
MlirType mooreEnumTypeGetWithBase(MlirAttribute name, MlirLocation loc,
                                  MlirType base) {
  return wrap(EnumType::get(unwrap(name).cast<StringAttr>(), unwrap(loc),
                            unwrap(base).cast<PackedType>()));
}

/// Create a simple bit-vector type.
MlirType mooreSimpleBitVectorTypeGet(MlirContext ctx, bool isFourValued,
                                     bool isSigned, unsigned size) {
  Domain domain = isFourValued ? Domain::FourValued : Domain::TwoValued;
  Sign sign = isSigned ? Sign::Signed : Sign::Unsigned;
  return wrap(SimpleBitVectorType(domain, sign, size).getType(unwrap(ctx)));
}

/// Checks whether the passed UnpackedType is a four-valued type.
bool mooreIsFourValuedType(MlirType type) {
  return unwrap(type).cast<UnpackedType>().getDomain() == Domain::FourValued;
}

/// Checks whether the passed type is a simple bit-vector.
bool mooreIsSimpleBitVectorType(MlirType type) {
  return unwrap(type).cast<UnpackedType>().isSimpleBitVector();
}

/// Returns the size of a simple bit-vector type in bits.
unsigned mooreGetSimpleBitVectorSize(MlirType type) {
  return unwrap(type).cast<UnpackedType>().getSimpleBitVector().size;
}

//===- FIRRTL.cpp - C Interface for the FIRRTL Dialect --------------------===//
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/FIRRTL.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(FIRRTL, firrtl,
                                      circt::firrtl::FIRRTLDialect)

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

MlirType firrtlTypeGetUInt(MlirContext ctx, int32_t width) {
  return wrap(UIntType::get(unwrap(ctx), width));
}

MlirType firrtlTypeGetSInt(MlirContext ctx, int32_t width) {
  return wrap(SIntType::get(unwrap(ctx), width));
}

MlirType firrtlTypeGetClock(MlirContext ctx) {
  return wrap(ClockType::get(unwrap(ctx)));
}

MlirType firrtlTypeGetReset(MlirContext ctx) {
  return wrap(ResetType::get(unwrap(ctx)));
}

MlirType firrtlTypeGetAsyncReset(MlirContext ctx) {
  return wrap(AsyncResetType::get(unwrap(ctx)));
}

MlirType firrtlTypeGetAnalog(MlirContext ctx, int32_t width) {
  return wrap(AnalogType::get(unwrap(ctx), width));
}

MlirType firrtlTypeGetVector(MlirContext ctx, MlirType element, size_t count) {
  auto baseType = unwrap(element).cast<FIRRTLBaseType>();
  assert(baseType && "element must be base type");

  return wrap(FVectorType::get(baseType, count));
}

MlirType firrtlTypeGetBundle(MlirContext ctx, size_t count,
                             const FIRRTLBundleField *fields) {
  SmallVector<BundleType::BundleElement, 4> bundleFields;
  bundleFields.reserve(count);

  for (size_t i = 0; i < count; i++) {
    auto field = fields[i];

    auto baseType = unwrap(field.type).dyn_cast<FIRRTLBaseType>();
    assert(baseType && "field must be base type");

    bundleFields.emplace_back(unwrap(field.name), field.isFlip, baseType);
  }
  return wrap(BundleType::get(unwrap(ctx), bundleFields));
}

//===----------------------------------------------------------------------===//
// Attribute API.
//===----------------------------------------------------------------------===//

MlirAttribute firrtlAttrGetConvention(MlirContext ctx,
                                      FIRRTLConvention convention) {
  Convention value;

  switch (convention) {
  case FIRRTL_CONVENTION_INTERNAL:
    value = Convention::Internal;
    break;
  case FIRRTL_CONVENTION_SCALARIZED:
    value = Convention::Scalarized;
    break;
  default: // NOLINT(clang-diagnostic-covered-switch-default)
    llvm_unreachable("unknown convention");
  }

  return wrap(ConventionAttr::get(unwrap(ctx), value));
}

MlirAttribute firrtlAttrGetPortDirs(MlirContext ctx, size_t count,
                                    const FIRRTLPortDir *dirs) {
  static_assert(FIRRTLPortDir::FIRRTL_PORT_DIR_INPUT ==
                static_cast<std::underlying_type_t<Direction>>(Direction::In));
  static_assert(FIRRTLPortDir::FIRRTL_PORT_DIR_OUTPUT ==
                static_cast<std::underlying_type_t<Direction>>(Direction::Out));

  // FIXME: The `reinterpret_cast` here may voilate strict aliasing rule. Is
  // there a better way?
  return wrap(direction::packAttribute(
      unwrap(ctx), ArrayRef(reinterpret_cast<const Direction *>(dirs), count)));
}

MlirAttribute firrtlAttrGetNameKind(MlirContext ctx, FIRRTLNameKind nameKind) {
  NameKindEnum value;

  switch (nameKind) {
  case FIRRTL_NAME_KIND_DROPPABLE_NAME:
    value = NameKindEnum::DroppableName;
    break;
  case FIRRTL_NAME_KIND_INTERESTING_NAME:
    value = NameKindEnum::InterestingName;
    break;
  default: // NOLINT(clang-diagnostic-covered-switch-default)
    llvm_unreachable("unknown name kind");
  }

  return wrap(NameKindEnumAttr::get(unwrap(ctx), value));
}

MlirAttribute firrtlAttrGetRUW(MlirContext ctx, FIRRTLRUW ruw) {
  RUWAttr value;

  switch (ruw) {
  case FIRRTL_RUW_UNDEFINED:
    value = RUWAttr::Undefined;
    break;
  case FIRRTL_RUW_OLD:
    value = RUWAttr::Old;
    break;
  case FIRRTL_RUW_NEW:
    value = RUWAttr::New;
    break;
  default: // NOLINT(clang-diagnostic-covered-switch-default)
    llvm_unreachable("unknown ruw");
  }

  return wrap(RUWAttrAttr::get(unwrap(ctx), value));
}

MlirAttribute firrtlAttrGetMemInit(MlirContext ctx, MlirIdentifier filename,
                                   bool isBinary, bool isInline) {
  return wrap(
      MemoryInitAttr::get(unwrap(ctx), unwrap(filename), isBinary, isInline));
}

MlirAttribute firrtlAttrGetMemDir(MlirContext ctx, FIRRTLMemDir dir) {
  MemDirAttr value;

  switch (dir) {
  case FIRRTL_MEM_DIR_INFER:
    value = MemDirAttr::Infer;
    break;
  case FIRRTL_MEM_DIR_READ:
    value = MemDirAttr::Read;
    break;
  case FIRRTL_MEM_DIR_WRITE:
    value = MemDirAttr::Write;
    break;
  case FIRRTL_MEM_DIR_READ_WRITE:
    value = MemDirAttr::ReadWrite;
    break;
  default: // NOLINT(clang-diagnostic-covered-switch-default)
    llvm_unreachable("unknown memory direction");
  }

  return wrap(MemDirAttrAttr::get(unwrap(ctx), value));
}

MlirAttribute firrtlAttrGetEventControl(MlirContext ctx,
                                        FIRRTLEventControl eventControl) {
  EventControl value;

  switch (eventControl) {
  case FIRRTL_EVENT_CONTROL_AT_POS_EDGE:
    value = EventControl::AtPosEdge;
    break;
  case FIRRTL_EVENT_CONTROL_AT_NEG_EDGE:
    value = EventControl::AtNegEdge;
    break;
  case FIRRTL_EVENT_CONTROL_AT_EDGE:
    value = EventControl::AtEdge;
    break;
  default: // NOLINT(clang-diagnostic-covered-switch-default)
    llvm_unreachable("unknown event control");
  }

  return wrap(EventControlAttr::get(unwrap(ctx), value));
}

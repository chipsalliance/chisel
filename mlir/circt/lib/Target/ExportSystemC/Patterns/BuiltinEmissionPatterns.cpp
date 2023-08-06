//===- BuiltinEmissionPatterns.cpp - Builtin Dialect Emission Patterns ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the emission patterns for the builtin dialect.
//
//===----------------------------------------------------------------------===//

#include "BuiltinEmissionPatterns.h"
#include "../EmissionPrinter.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallString.h"

using namespace mlir;
using namespace circt;
using namespace circt::ExportSystemC;

//===----------------------------------------------------------------------===//
// Operation emission patterns.
//===----------------------------------------------------------------------===//

namespace {

/// Emit the builtin module op by emitting all children in sequence. As a
/// result, we don't have to hard-code the behavior in ExportSytemC.
struct ModuleEmitter : OpEmissionPattern<ModuleOp> {
  using OpEmissionPattern::OpEmissionPattern;
  void emitStatement(ModuleOp op, EmissionPrinter &p) override {
    auto scope = p.getOstream().scope("", "", false);
    p.emitRegion(op.getRegion(), scope);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Type emission patterns.
//===----------------------------------------------------------------------===//

namespace {

/// Emit the builtin integer type to native C integer types.
struct IntegerTypeEmitter : TypeEmissionPattern<IntegerType> {
  bool match(Type type) override {
    if (!type.isa<IntegerType>())
      return false;

    unsigned bw = type.getIntOrFloatBitWidth();
    return bw == 1 || bw == 8 || bw == 16 || bw == 32 || bw == 64;
  }

  void emitType(IntegerType type, EmissionPrinter &p) override {
    unsigned bitWidth = type.getIntOrFloatBitWidth();
    switch (bitWidth) {
    case 1:
      p << "bool";
      break;
    case 8:
    case 16:
    case 32:
    case 64:
      p << (type.isSigned() ? "" : "u") << "int" << bitWidth << "_t";
      break;
    default:
      p.emitError("in the IntegerType emitter all cases allowed by the 'match' "
                  "function must be covered")
          << ", got uncovered case " << type;
    }
  }
};

/// Emit a builtin index type as 'size_t'.
struct IndexTypeEmitter : TypeEmissionPattern<IndexType> {
  void emitType(IndexType type, EmissionPrinter &p) override { p << "size_t"; }
};

} // namespace

namespace {

/// Emit a builtin string attribute as a C string literal including the
/// quotation marks.
struct StringAttrEmitter : AttrEmissionPattern<StringAttr> {
  void emitAttr(StringAttr attr, EmissionPrinter &p) override {
    attr.print(p.getOstream());
  }
};

/// Emit a builtin integer attribute as an integer literal. Integers with a
/// bitwidth of one are emitted as boolean literals 'true' and 'false'.
struct IntegerAttrEmitter : AttrEmissionPattern<IntegerAttr> {
  void emitAttr(IntegerAttr attr, EmissionPrinter &p) override {
    auto val = attr.getValue();

    if (val.getBitWidth() == 1) {
      p << (val.getBoolValue() ? "true" : "false");
    } else {
      bool isSigned = false;
      if (auto integer = attr.getType().dyn_cast<IntegerType>())
        isSigned = integer.isSigned();

      SmallString<128> strValue;
      val.toString(strValue, 10, isSigned);
      p << strValue;
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Register Operation and Type emission patterns.
//===----------------------------------------------------------------------===//

void circt::ExportSystemC::populateBuiltinOpEmitters(
    OpEmissionPatternSet &patterns, MLIRContext *context) {
  patterns.add<ModuleEmitter>(context);
}

void circt::ExportSystemC::populateBuiltinTypeEmitters(
    TypeEmissionPatternSet &patterns) {
  patterns.add<IntegerTypeEmitter, IndexTypeEmitter>();
}

void circt::ExportSystemC::populateBuiltinAttrEmitters(
    AttrEmissionPatternSet &patterns) {
  patterns.add<StringAttrEmitter, IntegerAttrEmitter>();
}

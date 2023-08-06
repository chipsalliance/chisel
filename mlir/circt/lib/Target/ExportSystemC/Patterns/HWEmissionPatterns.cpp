//===- HWEmissionPatterns.cpp - HW Dialect Emission Patterns --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the emission patterns for the HW dialect.
//
//===----------------------------------------------------------------------===//

#include "HWEmissionPatterns.h"
#include "../EmissionPrinter.h"
#include "circt/Dialect/HW/HWOps.h"

using namespace circt;
using namespace circt::hw;
using namespace circt::ExportSystemC;

//===----------------------------------------------------------------------===//
// Operation emission patterns.
//===----------------------------------------------------------------------===//

namespace {
/// The ConstantOp always inlines its value. Examples:
/// * hw.constant 5 : i32 ==> 5
/// * hw.constant 0 : i1 ==> false
/// * hw.constant 1 : i1 ==> true
struct ConstantEmitter : OpEmissionPattern<ConstantOp> {
  using OpEmissionPattern::OpEmissionPattern;

  MatchResult matchInlinable(Value value) override {
    if (value.getDefiningOp<ConstantOp>())
      return Precedence::LIT;
    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    p.emitAttr(value.getDefiningOp<ConstantOp>().getValueAttr());
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Register Operation and Type emission patterns.
//===----------------------------------------------------------------------===//

void circt::ExportSystemC::populateHWEmitters(OpEmissionPatternSet &patterns,
                                              MLIRContext *context) {
  patterns.add<ConstantEmitter>(context);
}

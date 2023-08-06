//===- LowerIntrinsics.cpp - Lower Intrinsics -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerIntrinsics pass.  This pass processes FIRRTL
// extmodules with intrinsic annotations and rewrites the instances as
// appropriate.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"

using namespace circt;
using namespace firrtl;

// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct LowerIntrinsicsPass : public LowerIntrinsicsBase<LowerIntrinsicsPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

static ParseResult hasNPorts(StringRef name, FModuleLike mod, unsigned n) {
  if (mod.getPorts().size() != n) {
    mod.emitError(name) << " has " << mod.getPorts().size()
                        << " ports instead of " << n;
    return failure();
  }
  return success();
}

static ParseResult namedPort(StringRef name, FModuleLike mod, unsigned n,
                             StringRef portName) {
  auto ports = mod.getPorts();
  if (n >= ports.size()) {
    mod.emitError(name) << " missing port " << n;
    return failure();
  }
  if (!ports[n].getName().equals(portName)) {
    mod.emitError(name) << " port " << n << " named '" << ports[n].getName()
                        << "' instead of '" << portName << "'";
    return failure();
  }
  return success();
}

template <typename T>
static ParseResult typedPort(StringRef name, FModuleLike mod, unsigned n) {
  auto ports = mod.getPorts();
  if (n >= ports.size()) {
    mod.emitError(name) << " missing port " << n;
    return failure();
  }
  if (!isa<T>(ports[n].type)) {
    mod.emitError(name) << " port " << n << " not of correct type";
    return failure();
  }
  return success();
}

template <typename T>
static ParseResult sizedPort(StringRef name, FModuleLike mod, unsigned n,
                             int32_t size) {
  auto ports = mod.getPorts();
  if (failed(typedPort<T>(name, mod, n)))
    return failure();
  if (cast<T>(ports[n].type).getWidth() != size) {
    mod.emitError(name) << " port " << n << " not size " << size;
    return failure();
  }
  return success();
}

static ParseResult resetPort(StringRef name, FModuleLike mod, unsigned n) {
  auto ports = mod.getPorts();
  if (n >= ports.size()) {
    mod.emitError(name) << " missing port " << n;
    return failure();
  }
  if (isa<ResetType, AsyncResetType>(ports[n].type))
    return success();
  if (auto uintType = dyn_cast<UIntType>(ports[n].type))
    if (uintType.getWidth() == 1)
      return success();
  mod.emitError(name) << " port " << n << " not of correct type";
  return failure();
}

static ParseResult hasNParam(StringRef name, FModuleLike mod, unsigned n,
                             unsigned c = 0) {
  unsigned num = 0;
  if (mod.getParameters())
    num = mod.getParameters().size();
  if (num < n || num > n + c) {
    auto d = mod.emitError(name) << " has " << num << " parameters instead of ";
    if (c == 0)
      d << n;
    else
      d << " between " << n << " and " << (n + c);
    return failure();
  }
  return success();
}

static ParseResult namedParam(StringRef name, FModuleLike mod,
                              StringRef paramName, bool optional = false) {
  for (auto a : mod.getParameters()) {
    auto param = cast<ParamDeclAttr>(a);
    if (param.getName().getValue().equals(paramName)) {
      if (isa<StringAttr>(param.getValue()))
        return success();

      mod.emitError(name) << " has parameter '" << param.getName()
                          << "' which should be a string but is not";
      return failure();
    }
  }
  if (optional)
    return success();
  mod.emitError(name) << " is missing parameter " << paramName;
  return failure();
}

static ParseResult namedIntParam(StringRef name, FModuleLike mod,
                                 StringRef paramName, bool optional = false) {
  for (auto a : mod.getParameters()) {
    auto param = cast<ParamDeclAttr>(a);
    if (param.getName().getValue().equals(paramName)) {
      if (isa<IntegerAttr>(param.getValue()))
        return success();

      mod.emitError(name) << " has parameter '" << param.getName()
                          << "' which should be an integer but is not";
      return failure();
    }
  }
  if (optional)
    return success();
  mod.emitError(name) << " is missing parameter " << paramName;
  return failure();
}

static bool lowerCirctSizeof(InstanceGraph &ig, FModuleLike mod) {
  auto ports = mod.getPorts();
  if (hasNPorts("circt.sizeof", mod, 2) ||
      namedPort("circt.sizeof", mod, 0, "i") ||
      namedPort("circt.sizeof", mod, 1, "size") ||
      sizedPort<UIntType>("circt.sizeof", mod, 1, 32) ||
      hasNParam("circt.sizeof", mod, 0))
    return false;

  for (auto *use : ig.lookup(mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto inputWire = builder.create<WireOp>(ports[0].type).getResult();
    inst.getResult(0).replaceAllUsesWith(inputWire);
    auto size = builder.create<SizeOfIntrinsicOp>(inputWire);
    inst.getResult(1).replaceAllUsesWith(size);
    inst.erase();
  }
  return true;
}

static bool lowerCirctIsX(InstanceGraph &ig, FModuleLike mod) {
  auto ports = mod.getPorts();
  if (hasNPorts("circt.isX", mod, 2) || namedPort("circt.isX", mod, 0, "i") ||
      namedPort("circt.isX", mod, 1, "found") ||
      sizedPort<UIntType>("circt.isX", mod, 1, 1) ||
      hasNParam("circt.isX", mod, 0))
    return false;

  for (auto *use : ig.lookup(mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto inputWire = builder.create<WireOp>(ports[0].type).getResult();
    inst.getResult(0).replaceAllUsesWith(inputWire);
    auto size = builder.create<IsXIntrinsicOp>(inputWire);
    inst.getResult(1).replaceAllUsesWith(size);
    inst.erase();
  }
  return true;
}

static bool lowerCirctPlusArgTest(InstanceGraph &ig, FModuleLike mod) {
  if (hasNPorts("circt.plusargs.test", mod, 1) ||
      namedPort("circt.plusargs.test", mod, 0, "found") ||
      sizedPort<UIntType>("circt.plusargs.test", mod, 0, 1) ||
      hasNParam("circt.plusargs.test", mod, 1) ||
      namedParam("circt.plusargs.test", mod, "FORMAT"))
    return false;

  auto param = cast<ParamDeclAttr>(mod.getParameters()[0]);
  for (auto *use : ig.lookup(mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto newop = builder.create<PlusArgsTestIntrinsicOp>(
        cast<StringAttr>(param.getValue()));
    inst.getResult(0).replaceAllUsesWith(newop);
    inst.erase();
  }
  return true;
}

static bool lowerCirctPlusArgValue(InstanceGraph &ig, FModuleLike mod) {
  if (hasNPorts("circt.plusargs.value", mod, 2) ||
      namedPort("circt.plusargs.value", mod, 0, "found") ||
      namedPort("circt.plusargs.value", mod, 1, "result") ||
      sizedPort<UIntType>("circt.plusargs.value", mod, 0, 1) ||
      hasNParam("circt.plusargs.value", mod, 1) ||
      namedParam("circt.plusargs.value", mod, "FORMAT"))
    return false;

  auto param = cast<ParamDeclAttr>(mod.getParameters()[0]);

  for (auto *use : ig.lookup(mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto newop = builder.create<PlusArgsValueIntrinsicOp>(
        inst.getResultTypes(), cast<StringAttr>(param.getValue()));
    inst.getResult(0).replaceAllUsesWith(newop.getFound());
    inst.getResult(1).replaceAllUsesWith(newop.getResult());
    inst.erase();
  }
  return true;
}

static bool lowerCirctClockGate(InstanceGraph &ig, FModuleLike mod) {
  if (hasNPorts("circt.clock_gate", mod, 3) ||
      namedPort("circt.clock_gate", mod, 0, "in") ||
      namedPort("circt.clock_gate", mod, 1, "en") ||
      namedPort("circt.clock_gate", mod, 2, "out") ||
      typedPort<ClockType>("circt.clock_gate", mod, 0) ||
      sizedPort<UIntType>("circt.clock_gate", mod, 1, 1) ||
      typedPort<ClockType>("circt.clock_gate", mod, 2) ||
      hasNParam("circt.clock_gate", mod, 0))
    return false;

  for (auto *use : ig.lookup(mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto in = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto en = builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(in);
    inst.getResult(1).replaceAllUsesWith(en);
    auto out = builder.create<ClockGateIntrinsicOp>(in, en, Value{});
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
  }
  return true;
}

template <bool isMux2>
static bool lowerCirctMuxCell(InstanceGraph &ig, FModuleLike mod) {
  StringRef mnemonic = isMux2 ? "circt.mux2cell" : "circt.mux4cell";
  unsigned portNum = isMux2 ? 4 : 6;
  if (hasNPorts(mnemonic, mod, portNum) || namedPort(mnemonic, mod, 0, "sel") ||
      typedPort<UIntType>(mnemonic, mod, 0)) {
    return false;
  }

  if (isMux2) {
    if (namedPort(mnemonic, mod, 1, "high") ||
        namedPort(mnemonic, mod, 2, "low") ||
        namedPort(mnemonic, mod, 3, "out"))
      return false;
  } else {
    if (namedPort(mnemonic, mod, 1, "v3") ||
        namedPort(mnemonic, mod, 2, "v2") ||
        namedPort(mnemonic, mod, 3, "v1") ||
        namedPort(mnemonic, mod, 4, "v0") || namedPort(mnemonic, mod, 5, "out"))
      return false;
  }

  for (auto *use : ig.lookup(mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    SmallVector<Value> operands;
    operands.reserve(portNum - 1);
    for (unsigned i = 0; i < portNum - 1; i++) {
      auto v = builder.create<WireOp>(inst.getResult(i).getType()).getResult();
      operands.push_back(v);
      inst.getResult(i).replaceAllUsesWith(v);
    }
    Value out;
    if (isMux2)
      out = builder.create<Mux2CellIntrinsicOp>(operands);
    else
      out = builder.create<Mux4CellIntrinsicOp>(operands);
    inst.getResult(portNum - 1).replaceAllUsesWith(out);
    inst.erase();
  }

  return true;
}

static bool lowerCirctLTLAnd(InstanceGraph &ig, FModuleLike mod) {
  if (hasNPorts("circt.ltl.and", mod, 3) ||
      namedPort("circt.ltl.and", mod, 0, "lhs") ||
      namedPort("circt.ltl.and", mod, 1, "rhs") ||
      namedPort("circt.ltl.and", mod, 2, "out") ||
      sizedPort<UIntType>("circt.ltl.and", mod, 0, 1) ||
      sizedPort<UIntType>("circt.ltl.and", mod, 1, 1) ||
      sizedPort<UIntType>("circt.ltl.and", mod, 2, 1) ||
      hasNParam("circt.ltl.and", mod, 0))
    return false;

  for (auto *use : ig.lookup(mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto lhs = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto rhs = builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(lhs);
    inst.getResult(1).replaceAllUsesWith(rhs);
    auto out = builder.create<LTLAndIntrinsicOp>(lhs.getType(), lhs, rhs);
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
  }
  return true;
}

static bool lowerCirctLTLOr(InstanceGraph &ig, FModuleLike mod) {
  if (hasNPorts("circt.ltl.or", mod, 3) ||
      namedPort("circt.ltl.or", mod, 0, "lhs") ||
      namedPort("circt.ltl.or", mod, 1, "rhs") ||
      namedPort("circt.ltl.or", mod, 2, "out") ||
      sizedPort<UIntType>("circt.ltl.or", mod, 0, 1) ||
      sizedPort<UIntType>("circt.ltl.or", mod, 1, 1) ||
      sizedPort<UIntType>("circt.ltl.or", mod, 2, 1) ||
      hasNParam("circt.ltl.or", mod, 0))
    return false;

  for (auto *use : ig.lookup(mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto lhs = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto rhs = builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(lhs);
    inst.getResult(1).replaceAllUsesWith(rhs);
    auto out = builder.create<LTLOrIntrinsicOp>(lhs.getType(), lhs, rhs);
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
  }
  return true;
}

static bool lowerCirctLTLDelay(InstanceGraph &ig, FModuleLike mod) {
  if (hasNPorts("circt.ltl.delay", mod, 2) ||
      namedPort("circt.ltl.delay", mod, 0, "in") ||
      namedPort("circt.ltl.delay", mod, 1, "out") ||
      sizedPort<UIntType>("circt.ltl.delay", mod, 0, 1) ||
      sizedPort<UIntType>("circt.ltl.delay", mod, 1, 1) ||
      hasNParam("circt.ltl.delay", mod, 1, 2) ||
      namedIntParam("circt.ltl.delay", mod, "delay") ||
      namedIntParam("circt.ltl.delay", mod, "length", true))
    return false;

  auto getI64Attr = [&](int64_t value) {
    return IntegerAttr::get(IntegerType::get(mod.getContext(), 64), value);
  };
  auto params = mod.getParameters();
  auto delay = getI64Attr(params[0]
                              .cast<ParamDeclAttr>()
                              .getValue()
                              .cast<IntegerAttr>()
                              .getValue()
                              .getZExtValue());
  IntegerAttr length;
  if (params.size() >= 2)
    if (auto lengthDecl = cast<ParamDeclAttr>(params[1]))
      length = getI64Attr(
          cast<IntegerAttr>(lengthDecl.getValue()).getValue().getZExtValue());

  for (auto *use : ig.lookup(mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto in = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(in);
    auto out =
        builder.create<LTLDelayIntrinsicOp>(in.getType(), in, delay, length);
    inst.getResult(1).replaceAllUsesWith(out);
    inst.erase();
  }
  return true;
}

static bool lowerCirctLTLConcat(InstanceGraph &ig, FModuleLike mod) {
  if (hasNPorts("circt.ltl.concat", mod, 3) ||
      namedPort("circt.ltl.concat", mod, 0, "lhs") ||
      namedPort("circt.ltl.concat", mod, 1, "rhs") ||
      namedPort("circt.ltl.concat", mod, 2, "out") ||
      sizedPort<UIntType>("circt.ltl.concat", mod, 0, 1) ||
      sizedPort<UIntType>("circt.ltl.concat", mod, 1, 1) ||
      sizedPort<UIntType>("circt.ltl.concat", mod, 2, 1) ||
      hasNParam("circt.ltl.concat", mod, 0))
    return false;

  for (auto *use : ig.lookup(mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto lhs = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto rhs = builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(lhs);
    inst.getResult(1).replaceAllUsesWith(rhs);
    auto out = builder.create<LTLConcatIntrinsicOp>(lhs.getType(), lhs, rhs);
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
  }
  return true;
}

static bool lowerCirctLTLNot(InstanceGraph &ig, FModuleLike mod) {
  if (hasNPorts("circt.ltl.not", mod, 2) ||
      namedPort("circt.ltl.not", mod, 0, "in") ||
      namedPort("circt.ltl.not", mod, 1, "out") ||
      sizedPort<UIntType>("circt.ltl.not", mod, 0, 1) ||
      sizedPort<UIntType>("circt.ltl.not", mod, 1, 1) ||
      hasNParam("circt.ltl.not", mod, 0))
    return false;

  for (auto *use : ig.lookup(mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto input =
        builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(input);
    auto out = builder.create<LTLNotIntrinsicOp>(input.getType(), input);
    inst.getResult(1).replaceAllUsesWith(out);
    inst.erase();
  }
  return true;
}

static bool lowerCirctLTLImplication(InstanceGraph &ig, FModuleLike mod) {
  if (hasNPorts("circt.ltl.implication", mod, 3) ||
      namedPort("circt.ltl.implication", mod, 0, "lhs") ||
      namedPort("circt.ltl.implication", mod, 1, "rhs") ||
      namedPort("circt.ltl.implication", mod, 2, "out") ||
      sizedPort<UIntType>("circt.ltl.implication", mod, 0, 1) ||
      sizedPort<UIntType>("circt.ltl.implication", mod, 1, 1) ||
      sizedPort<UIntType>("circt.ltl.implication", mod, 2, 1) ||
      hasNParam("circt.ltl.implication", mod, 0))
    return false;

  for (auto *use : ig.lookup(mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto lhs = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto rhs = builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(lhs);
    inst.getResult(1).replaceAllUsesWith(rhs);
    auto out =
        builder.create<LTLImplicationIntrinsicOp>(lhs.getType(), lhs, rhs);
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
  }
  return true;
}

static bool lowerCirctLTLEventually(InstanceGraph &ig, FModuleLike mod) {
  if (hasNPorts("circt.ltl.eventually", mod, 2) ||
      namedPort("circt.ltl.eventually", mod, 0, "in") ||
      namedPort("circt.ltl.eventually", mod, 1, "out") ||
      sizedPort<UIntType>("circt.ltl.eventually", mod, 0, 1) ||
      sizedPort<UIntType>("circt.ltl.eventually", mod, 1, 1) ||
      hasNParam("circt.ltl.eventually", mod, 0))
    return false;

  for (auto *use : ig.lookup(mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto input =
        builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(input);
    auto out = builder.create<LTLEventuallyIntrinsicOp>(input.getType(), input);
    inst.getResult(1).replaceAllUsesWith(out);
    inst.erase();
  }
  return true;
}

static bool lowerCirctLTLClock(InstanceGraph &ig, FModuleLike mod) {
  if (hasNPorts("circt.ltl.clock", mod, 3) ||
      namedPort("circt.ltl.clock", mod, 0, "in") ||
      namedPort("circt.ltl.clock", mod, 1, "clock") ||
      namedPort("circt.ltl.clock", mod, 2, "out") ||
      sizedPort<UIntType>("circt.ltl.clock", mod, 0, 1) ||
      typedPort<ClockType>("circt.ltl.clock", mod, 1) ||
      sizedPort<UIntType>("circt.ltl.clock", mod, 2, 1) ||
      hasNParam("circt.ltl.clock", mod, 0))
    return false;

  for (auto *use : ig.lookup(mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto in = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto clock =
        builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(in);
    inst.getResult(1).replaceAllUsesWith(clock);
    auto out = builder.create<LTLClockIntrinsicOp>(in.getType(), in, clock);
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
  }
  return true;
}

static bool lowerCirctLTLDisable(InstanceGraph &ig, FModuleLike mod) {
  if (hasNPorts("circt.ltl.disable", mod, 3) ||
      namedPort("circt.ltl.disable", mod, 0, "in") ||
      namedPort("circt.ltl.disable", mod, 1, "condition") ||
      namedPort("circt.ltl.disable", mod, 2, "out") ||
      sizedPort<UIntType>("circt.ltl.disable", mod, 0, 1) ||
      sizedPort<UIntType>("circt.ltl.disable", mod, 1, 1) ||
      sizedPort<UIntType>("circt.ltl.disable", mod, 2, 1) ||
      hasNParam("circt.ltl.disable", mod, 0))
    return false;

  for (auto *use : ig.lookup(mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto in = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto condition =
        builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(in);
    inst.getResult(1).replaceAllUsesWith(condition);
    auto out =
        builder.create<LTLDisableIntrinsicOp>(in.getType(), in, condition);
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
  }
  return true;
}

template <class Op>
static bool lowerCirctVerif(InstanceGraph &ig, FModuleLike mod) {
  if (hasNPorts("circt.verif.assert", mod, 1) ||
      namedPort("circt.verif.assert", mod, 0, "property") ||
      sizedPort<UIntType>("circt.verif.assert", mod, 0, 1) ||
      hasNParam("circt.verif.assert", mod, 0, 1) ||
      namedParam("circt.verif.assert", mod, "label", true))
    return false;

  auto params = mod.getParameters();
  StringAttr label;
  if (!params.empty())
    if (auto labelDecl = cast<ParamDeclAttr>(params[0]))
      label = cast<StringAttr>(labelDecl.getValue());

  for (auto *use : ig.lookup(mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto property =
        builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(property);
    builder.create<Op>(property, label);
    inst.erase();
  }
  return true;
}

static bool lowerCirctHasBeenReset(InstanceGraph &ig, FModuleLike mod) {
  if (hasNPorts("circt.has_been_reset", mod, 3) ||
      namedPort("circt.has_been_reset", mod, 0, "clock") ||
      namedPort("circt.has_been_reset", mod, 1, "reset") ||
      namedPort("circt.has_been_reset", mod, 2, "out") ||
      typedPort<ClockType>("circt.has_been_reset", mod, 0) ||
      resetPort("circt.has_been_reset", mod, 1) ||
      sizedPort<UIntType>("circt.has_been_reset", mod, 2, 1) ||
      hasNParam("circt.has_been_reset", mod, 0))
    return false;

  for (auto *use : ig.lookup(mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto clock =
        builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto reset =
        builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(clock);
    inst.getResult(1).replaceAllUsesWith(reset);
    auto out = builder.create<HasBeenResetIntrinsicOp>(clock, reset);
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
  }
  return true;
}

std::pair<const char *, std::function<bool(InstanceGraph &, FModuleLike)>>
    intrinsics[] = {
        {"circt.sizeof", lowerCirctSizeof},
        {"circt_sizeof", lowerCirctSizeof},
        {"circt.isX", lowerCirctIsX},
        {"circt_isX", lowerCirctIsX},
        {"circt.plusargs.test", lowerCirctPlusArgTest},
        {"circt_plusargs_test", lowerCirctPlusArgTest},
        {"circt.plusargs.value", lowerCirctPlusArgValue},
        {"circt_plusargs_value", lowerCirctPlusArgValue},
        {"circt.clock_gate", lowerCirctClockGate},
        {"circt_clock_gate", lowerCirctClockGate},
        {"circt.ltl.and", lowerCirctLTLAnd},
        {"circt_ltl_and", lowerCirctLTLAnd},
        {"circt.ltl.or", lowerCirctLTLOr},
        {"circt_ltl_or", lowerCirctLTLOr},
        {"circt.ltl.delay", lowerCirctLTLDelay},
        {"circt_ltl_delay", lowerCirctLTLDelay},
        {"circt.ltl.concat", lowerCirctLTLConcat},
        {"circt_ltl_concat", lowerCirctLTLConcat},
        {"circt.ltl.not", lowerCirctLTLNot},
        {"circt_ltl_not", lowerCirctLTLNot},
        {"circt.ltl.implication", lowerCirctLTLImplication},
        {"circt_ltl_implication", lowerCirctLTLImplication},
        {"circt.ltl.eventually", lowerCirctLTLEventually},
        {"circt_ltl_eventually", lowerCirctLTLEventually},
        {"circt.ltl.clock", lowerCirctLTLClock},
        {"circt_ltl_clock", lowerCirctLTLClock},
        {"circt.ltl.disable", lowerCirctLTLDisable},
        {"circt_ltl_disable", lowerCirctLTLDisable},
        {"circt.verif.assert", lowerCirctVerif<VerifAssertIntrinsicOp>},
        {"circt_verif_assert", lowerCirctVerif<VerifAssertIntrinsicOp>},
        {"circt.verif.assume", lowerCirctVerif<VerifAssumeIntrinsicOp>},
        {"circt_verif_assume", lowerCirctVerif<VerifAssumeIntrinsicOp>},
        {"circt.verif.cover", lowerCirctVerif<VerifCoverIntrinsicOp>},
        {"circt_verif_cover", lowerCirctVerif<VerifCoverIntrinsicOp>},
        {"circt.mux2cell", lowerCirctMuxCell<true>},
        {"circt_mux2cell", lowerCirctMuxCell<true>},
        {"circt.mux4cell", lowerCirctMuxCell<false>},
        {"circt_mux4cell", lowerCirctMuxCell<false>},
        {"circt.has_been_reset", lowerCirctHasBeenReset},
        {"circt_has_been_reset", lowerCirctHasBeenReset}};

// This is the main entrypoint for the lowering pass.
void LowerIntrinsicsPass::runOnOperation() {
  size_t numFailures = 0;
  size_t numConverted = 0;
  InstanceGraph &ig = getAnalysis<InstanceGraph>();
  for (auto &op : llvm::make_early_inc_range(getOperation().getOps())) {
    if (!isa<FExtModuleOp, FIntModuleOp>(op))
      continue;
    StringAttr intname;
    if (isa<FExtModuleOp>(op)) {
      auto anno = AnnotationSet(&op).getAnnotation("circt.Intrinsic");
      if (!anno)
        continue;
      intname = anno.getMember<StringAttr>("intrinsic");
      if (!intname) {
        op.emitError("intrinsic annotation with no intrinsic name");
        ++numFailures;
        continue;
      }
    } else {
      intname = cast<FIntModuleOp>(op).getIntrinsicAttr();
      if (!intname) {
        op.emitError("intrinsic module with no intrinsic name");
        ++numFailures;
        continue;
      }
    }

    bool found = false;
    for (const auto &intrinsic : intrinsics) {
      if (intname.getValue().equals(intrinsic.first)) {
        found = true;
        if (intrinsic.second(ig, cast<FModuleLike>(op))) {
          ++numConverted;
          op.erase();
        } else {
          ++numFailures;
        }
        break;
      }
    }
    if (!found) {
      op.emitError("unknown intrinsic: '") << intname.getValue() << "'";
      ++numFailures;
    }
  }
  if (numFailures)
    signalPassFailure();
  if (!numConverted)
    markAllAnalysesPreserved();
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createLowerIntrinsicsPass() {
  return std::make_unique<LowerIntrinsicsPass>();
}

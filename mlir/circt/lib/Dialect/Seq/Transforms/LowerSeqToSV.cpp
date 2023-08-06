//===- LowerSeqToSV.cpp - Seq to SV lowering ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transform translate Seq ops to SV.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/Namespace.h"
#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace seq {
#define GEN_PASS_DEF_LOWERSEQFIRRTLINITTOSV
#include "circt/Dialect/Seq/SeqPasses.h.inc"
} // namespace seq
} // namespace circt

using namespace circt;
using namespace seq;

namespace {
#define GEN_PASS_DEF_LOWERSEQTOSV
#define GEN_PASS_DEF_LOWERSEQFIRRTLTOSV
#define GEN_PASS_DEF_LOWERSEQFIRRTLINITTOSV
#include "circt/Dialect/Seq/SeqPasses.h.inc"

struct SeqToSVPass : public impl::LowerSeqToSVBase<SeqToSVPass> {
  void runOnOperation() override;
  using LowerSeqToSVBase::lowerToAlwaysFF;
};

struct SeqFIRRTLToSVPass
    : public impl::LowerSeqFIRRTLToSVBase<SeqFIRRTLToSVPass> {
  void runOnOperation() override;
  using LowerSeqFIRRTLToSVBase<SeqFIRRTLToSVPass>::disableRegRandomization;
  using LowerSeqFIRRTLToSVBase<SeqFIRRTLToSVPass>::emitSeparateAlwaysBlocks;
  using LowerSeqFIRRTLToSVBase<SeqFIRRTLToSVPass>::LowerSeqFIRRTLToSVBase;
  using LowerSeqFIRRTLToSVBase<SeqFIRRTLToSVPass>::numSubaccessRestored;
};
struct SeqFIRRTLInitToSVPass
    : public impl::LowerSeqFIRRTLInitToSVBase<SeqFIRRTLInitToSVPass> {
  void runOnOperation() override;
};
} // anonymous namespace

/// Create the assign.
static void createAssign(ConversionPatternRewriter &rewriter, sv::RegOp svReg,
                         CompRegOp reg) {
  rewriter.create<sv::PAssignOp>(reg.getLoc(), svReg, reg.getInput());
}
/// Create the assign inside of an if block.
static void createAssign(ConversionPatternRewriter &rewriter, sv::RegOp svReg,
                         CompRegClockEnabledOp reg) {
  Location loc = reg.getLoc();
  rewriter.create<sv::IfOp>(loc, reg.getClockEnable(), [&]() {
    rewriter.create<sv::PAssignOp>(reg.getLoc(), svReg, reg.getInput());
  });
}

namespace {
/// Lower CompRegOp to `sv.reg` and `sv.alwaysff`. Use a posedge clock and
/// synchronous reset.
template <typename OpTy>
class CompRegLower : public OpConversionPattern<OpTy> {
public:
  CompRegLower(MLIRContext *context, bool lowerToAlwaysFF)
      : OpConversionPattern<OpTy>(context), lowerToAlwaysFF(lowerToAlwaysFF) {}

  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<OpTy>::OpAdaptor;

  LogicalResult
  matchAndRewrite(OpTy reg, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = reg.getLoc();

    auto svReg =
        rewriter.create<sv::RegOp>(loc, reg.getResult().getType(),
                                   reg.getNameAttr(), reg.getInnerSymAttr());
    svReg->setDialectAttrs(reg->getDialectAttrs());

    circt::sv::setSVAttributes(svReg, circt::sv::getSVAttributes(reg));

    auto regVal = rewriter.create<sv::ReadInOutOp>(loc, svReg);

    auto assignValue = [&] { createAssign(rewriter, svReg, reg); };
    auto assignReset = [&] {
      rewriter.create<sv::PAssignOp>(loc, svReg, adaptor.getResetValue());
    };

    if (adaptor.getReset() && adaptor.getResetValue()) {
      if (lowerToAlwaysFF) {
        rewriter.create<sv::AlwaysFFOp>(
            loc, sv::EventControl::AtPosEdge, adaptor.getClk(),
            ResetType::SyncReset, sv::EventControl::AtPosEdge,
            adaptor.getReset(), assignValue, assignReset);
      } else {
        rewriter.create<sv::AlwaysOp>(
            loc, sv::EventControl::AtPosEdge, adaptor.getClk(), [&] {
              rewriter.create<sv::IfOp>(loc, adaptor.getReset(), assignReset,
                                        assignValue);
            });
      }
    } else {
      if (lowerToAlwaysFF) {
        rewriter.create<sv::AlwaysFFOp>(loc, sv::EventControl::AtPosEdge,
                                        adaptor.getClk(), assignValue);
      } else {
        rewriter.create<sv::AlwaysOp>(loc, sv::EventControl::AtPosEdge,
                                      adaptor.getClk(), assignValue);
      }
    }

    rewriter.replaceOp(reg, regVal);
    return success();
  }

private:
  bool lowerToAlwaysFF;
};

// Lower seq.clock_gate to a fairly standard clock gate implementation.
//
class ClockGateLowering : public OpConversionPattern<ClockGateOp> {
public:
  using OpConversionPattern<ClockGateOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<ClockGateOp>::OpAdaptor;
  LogicalResult
  matchAndRewrite(ClockGateOp clockGate, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = clockGate.getLoc();
    Value clk = adaptor.getInput();

    // enable in
    Value enable = adaptor.getEnable();
    if (auto te = adaptor.getTestEnable())
      enable = rewriter.create<comb::OrOp>(loc, enable, te);

    // Enable latch.
    Value enableLatch = rewriter.create<sv::RegOp>(
        loc, rewriter.getI1Type(), rewriter.getStringAttr("cg_en_latch"));

    // Latch the enable signal using an always @* block.
    rewriter.create<sv::AlwaysOp>(
        loc, llvm::SmallVector<sv::EventControl>{}, llvm::SmallVector<Value>{},
        [&]() {
          rewriter.create<sv::IfOp>(
              loc, comb::createOrFoldNot(loc, clk, rewriter), [&]() {
                rewriter.create<sv::PAssignOp>(loc, enableLatch, enable);
              });
        });

    // Create the gated clock signal.
    Value gclk = rewriter.create<comb::AndOp>(
        loc, clk, rewriter.create<sv::ReadInOutOp>(loc, enableLatch));
    clockGate.replaceAllUsesWith(gclk);
    rewriter.eraseOp(clockGate);
    return success();
  }
};

} // namespace

namespace {
/// Lower FirRegOp to `sv.reg` and `sv.always`.
class FirRegLower {
public:
  FirRegLower(hw::HWModuleOp module, bool disableRegRandomization = false,
              bool emitSeparateAlwaysBlocks = false)
      : module(module), disableRegRandomization(disableRegRandomization),
        emitSeparateAlwaysBlocks(emitSeparateAlwaysBlocks){};

  void lower();

  unsigned numSubaccessRestored = 0;

private:
  struct RegLowerInfo {
    sv::RegOp reg;
    IntegerAttr preset;
    Value asyncResetSignal;
    Value asyncResetValue;
    int64_t randStart;
    size_t width;
  };

  RegLowerInfo lower(FirRegOp reg);

  void initialize(OpBuilder &builder, RegLowerInfo reg, ArrayRef<Value> rands);
  void initializeRegisterElements(Location loc, OpBuilder &builder, Value reg,
                                  Value rand, unsigned &pos);

  void createTree(OpBuilder &builder, Value reg, Value term, Value next);
  std::optional<std::tuple<Value, Value, Value>>
  tryRestoringSubaccess(OpBuilder &builder, Value reg, Value term,
                        hw::ArrayCreateOp nextArray);

  void addToAlwaysBlock(Block *block, sv::EventControl clockEdge, Value clock,
                        std::function<void(OpBuilder &)> body,
                        ResetType resetStyle = {},
                        sv::EventControl resetEdge = {}, Value reset = {},
                        std::function<void(OpBuilder &)> resetBody = {});

  void addToIfBlock(OpBuilder &builder, Value cond,
                    const std::function<void()> &trueSide,
                    const std::function<void()> &falseSide);

  hw::ConstantOp getOrCreateConstant(Location loc, const APInt &value) {
    OpBuilder builder(module.getBody());
    auto &constant = constantCache[value];
    if (constant) {
      constant->setLoc(builder.getFusedLoc({constant->getLoc(), loc}));
      return constant;
    }

    constant = builder.create<hw::ConstantOp>(loc, value);
    return constant;
  }

  using AlwaysKeyType = std::tuple<Block *, sv::EventControl, Value, ResetType,
                                   sv::EventControl, Value>;
  llvm::SmallDenseMap<AlwaysKeyType, std::pair<sv::AlwaysOp, sv::IfOp>>
      alwaysBlocks;

  using IfKeyType = std::pair<Block *, Value>;
  llvm::SmallDenseMap<IfKeyType, sv::IfOp> ifCache;

  llvm::SmallDenseMap<APInt, hw::ConstantOp> constantCache;
  llvm::SmallDenseMap<std::pair<Value, unsigned>, Value> arrayIndexCache;

  hw::HWModuleOp module;

  bool disableRegRandomization;
  bool emitSeparateAlwaysBlocks;
};
} // namespace

void FirRegLower::addToIfBlock(OpBuilder &builder, Value cond,
                               const std::function<void()> &trueSide,
                               const std::function<void()> &falseSide) {
  auto op = ifCache.lookup({builder.getBlock(), cond});
  // Always build both sides of the if, in case we want to use an empty else
  // later. This way we don't have to build a new if and replace it.
  if (!op) {
    auto newIfOp =
        builder.create<sv::IfOp>(cond.getLoc(), cond, trueSide, falseSide);
    ifCache.insert({{builder.getBlock(), cond}, newIfOp});
  } else {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(op.getThenBlock());
    trueSide();
    builder.setInsertionPointToEnd(op.getElseBlock());
    falseSide();
  }
}

void FirRegLower::lower() {
  // Find all registers to lower in the module.
  auto regs = module.getOps<seq::FirRegOp>();
  if (regs.empty())
    return;

  // Lower the regs to SV regs. Group them by initializer and reset kind.
  SmallVector<RegLowerInfo> randomInit, presetInit;
  llvm::MapVector<Value, SmallVector<RegLowerInfo>> asyncResets;
  for (auto reg : llvm::make_early_inc_range(regs)) {
    auto svReg = lower(reg);
    if (svReg.preset)
      presetInit.push_back(svReg);
    else if (!disableRegRandomization)
      randomInit.push_back(svReg);

    if (svReg.asyncResetSignal)
      asyncResets[svReg.asyncResetSignal].emplace_back(svReg);
  }

  // Compute total width of random space.  Place non-chisel registers at the end
  // of the space.  The Random space is unique to the initial block, due to
  // verilog thread rules, so we can drop trailing random calls if they are
  // unused.
  uint64_t maxBit = 0;
  for (auto reg : randomInit)
    if (reg.randStart >= 0)
      maxBit = std::max(maxBit, (uint64_t)reg.randStart + reg.width);

  for (auto &reg : randomInit) {
    if (reg.randStart == -1) {
      reg.randStart = maxBit;
      maxBit += reg.width;
    }
  }

  // Create an initial block at the end of the module where random
  // initialisation will be inserted.  Create two builders into the two
  // `ifdef` ops where the registers will be placed.
  //
  // `ifndef SYNTHESIS
  //   `ifdef RANDOMIZE_REG_INIT
  //      ... regBuilder ...
  //   `endif
  //   initial
  //     `INIT_RANDOM_PROLOG_
  //     ... initBuilder ..
  // `endif
  if (randomInit.empty() && presetInit.empty() && asyncResets.empty())
    return;

  auto loc = module.getLoc();
  MLIRContext *context = module.getContext();
  auto randInitRef = sv::MacroIdentAttr::get(context, "RANDOMIZE_REG_INIT");

  auto builder =
      ImplicitLocOpBuilder::atBlockTerminator(loc, module.getBodyBlock());

  builder.create<sv::IfDefOp>("ENABLE_INITIAL_REG_", [&] {
    builder.create<sv::OrderedOutputOp>([&] {
      builder.create<sv::IfDefOp>("FIRRTL_BEFORE_INITIAL", [&] {
        builder.create<sv::VerbatimOp>("`FIRRTL_BEFORE_INITIAL");
      });

      builder.create<sv::InitialOp>([&] {
        if (!randomInit.empty()) {
          builder.create<sv::IfDefProceduralOp>("INIT_RANDOM_PROLOG_", [&] {
            builder.create<sv::VerbatimOp>("`INIT_RANDOM_PROLOG_");
          });
          builder.create<sv::IfDefProceduralOp>(randInitRef, [&] {
            // Create randomization vector
            SmallVector<Value> randValues;
            auto numRandomCalls = (maxBit + 31) / 32;
            auto logic = builder.create<sv::LogicOp>(
                loc,
                hw::UnpackedArrayType::get(builder.getIntegerType(32),
                                           numRandomCalls),
                "_RANDOM");
            // Indvar's width must be equal to `ceil(log2(numRandomCalls +
            // 1))` to avoid overflow.
            auto inducionVariableWidth = llvm::Log2_64_Ceil(numRandomCalls + 1);
            auto arrayIndexWith = llvm::Log2_64_Ceil(numRandomCalls);
            auto lb =
                getOrCreateConstant(loc, APInt::getZero(inducionVariableWidth));
            auto ub = getOrCreateConstant(
                loc, APInt(inducionVariableWidth, numRandomCalls));
            auto step =
                getOrCreateConstant(loc, APInt(inducionVariableWidth, 1));
            auto forLoop = builder.create<sv::ForOp>(
                loc, lb, ub, step, "i", [&](BlockArgument iter) {
                  auto rhs = builder.create<sv::MacroRefExprSEOp>(
                      loc, builder.getIntegerType(32), "RANDOM");
                  Value iterValue = iter;
                  if (!iter.getType().isInteger(arrayIndexWith))
                    iterValue = builder.create<comb::ExtractOp>(
                        loc, iterValue, 0, arrayIndexWith);
                  auto lhs = builder.create<sv::ArrayIndexInOutOp>(loc, logic,
                                                                   iterValue);
                  builder.create<sv::BPAssignOp>(loc, lhs, rhs);
                });
            builder.setInsertionPointAfter(forLoop);
            for (uint64_t x = 0; x < numRandomCalls; ++x) {
              auto lhs = builder.create<sv::ArrayIndexInOutOp>(
                  loc, logic,
                  getOrCreateConstant(loc, APInt(arrayIndexWith, x)));
              randValues.push_back(lhs.getResult());
            }

            // Create initialisers for all registers.
            for (auto &svReg : randomInit)
              initialize(builder, svReg, randValues);
          });
        }

        if (!presetInit.empty()) {
          for (auto &svReg : presetInit) {
            auto loc = svReg.reg.getLoc();
            auto cst = getOrCreateConstant(loc, svReg.preset.getValue());
            builder.create<sv::BPAssignOp>(loc, svReg.reg, cst);
          }
        }

        if (!asyncResets.empty()) {
          // If the register is async reset, we need to insert extra
          // initialization in post-randomization so that we can set the
          // reset value to register if the reset signal is enabled.
          for (auto &reset : asyncResets) {
            //  if (reset) begin
            //    ..
            //  end
            builder.create<sv::IfOp>(reset.first, [&] {
              for (auto &reg : reset.second)
                builder.create<sv::BPAssignOp>(reg.reg.getLoc(), reg.reg,
                                               reg.asyncResetValue);
            });
          }
        }
      });

      builder.create<sv::IfDefOp>("FIRRTL_AFTER_INITIAL", [&] {
        builder.create<sv::VerbatimOp>("`FIRRTL_AFTER_INITIAL");
      });
    });
  });

  module->removeAttr("firrtl.random_init_width");
}

// Return true if two arguments are equivalent, or if both of them are the same
// array indexing.
// NOLINTNEXTLINE(misc-no-recursion)
static bool areEquivalentValues(Value term, Value next) {
  if (term == next)
    return true;
  // Check whether these values are equivalent array accesses with constant
  // index. We have to check the equivalence recursively because they might not
  // be CSEd.
  if (auto t1 = term.getDefiningOp<hw::ArrayGetOp>())
    if (auto t2 = next.getDefiningOp<hw::ArrayGetOp>())
      if (auto c1 = t1.getIndex().getDefiningOp<hw::ConstantOp>())
        if (auto c2 = t2.getIndex().getDefiningOp<hw::ConstantOp>())
          return c1.getType() == c2.getType() &&
                 c1.getValue() == c2.getValue() &&
                 areEquivalentValues(t1.getInput(), t2.getInput());
  // Otherwise, regard as different.
  // TODO: Handle struct if necessary.
  return false;
}

static llvm::SetVector<Value> extractConditions(Value value) {
  auto andOp = value.getDefiningOp<comb::AndOp>();
  // If the value is not AndOp with a bin flag, use it as a condition.
  if (!andOp || !andOp.getTwoState()) {
    llvm::SetVector<Value> ret;
    ret.insert(value);
    return ret;
  }

  return llvm::SetVector<Value>(andOp.getOperands().begin(),
                                andOp.getOperands().end());
}

static std::optional<APInt> getConstantValue(Value value) {
  auto constantIndex = value.template getDefiningOp<hw::ConstantOp>();
  if (constantIndex)
    return constantIndex.getValue();
  return {};
}

// Return a tuple <cond, idx, val> if the array register update can be
// represented with a dynamic index assignment:
// if (cond)
//   reg[idx] <= val;
//
std::optional<std::tuple<Value, Value, Value>>
FirRegLower::tryRestoringSubaccess(OpBuilder &builder, Value reg, Value term,
                                   hw::ArrayCreateOp nextRegValue) {
  Value trueVal;
  SmallVector<Value> muxConditions;
  // Compat fix for GCC12's libstdc++, cannot use
  // llvm::enumerate(llvm::reverse(OperandRange)).  See #4900.
  SmallVector<Value> reverseOpValues(llvm::reverse(nextRegValue.getOperands()));
  if (!llvm::all_of(llvm::enumerate(reverseOpValues), [&](auto idxAndValue) {
        // Check that `nextRegValue[i]` is `cond_i ? val : reg[i]`.
        auto [i, value] = idxAndValue;
        auto mux = value.template getDefiningOp<comb::MuxOp>();
        // Ensure that mux has binary flag.
        if (!mux || !mux.getTwoState())
          return false;
        // The next value must be same.
        if (trueVal && trueVal != mux.getTrueValue())
          return false;
        if (!trueVal)
          trueVal = mux.getTrueValue();
        muxConditions.push_back(mux.getCond());
        // Check that ith element is an element of the register we are
        // currently lowering.
        auto arrayGet =
            mux.getFalseValue().template getDefiningOp<hw::ArrayGetOp>();
        if (!arrayGet)
          return false;
        return areEquivalentValues(arrayGet.getInput(), term) &&
               getConstantValue(arrayGet.getIndex()) == i;
      }))
    return {};

  // Extract common expressions among mux conditions.
  llvm::SetVector<Value> commonConditions =
      extractConditions(muxConditions.front());
  for (auto condition : ArrayRef(muxConditions).drop_front()) {
    auto cond = extractConditions(condition);
    commonConditions.remove_if([&](auto v) { return !cond.contains(v); });
  }
  Value indexValue;
  for (auto [idx, condition] : llvm::enumerate(muxConditions)) {
    llvm::SetVector<Value> extractedConditions = extractConditions(condition);
    // Remove common conditions and check the remaining condition is only an
    // index comparision.
    extractedConditions.remove_if(
        [&](auto v) { return commonConditions.contains(v); });
    if (extractedConditions.size() != 1)
      return {};

    auto indexCompare =
        (*extractedConditions.begin()).getDefiningOp<comb::ICmpOp>();
    if (!indexCompare || !indexCompare.getTwoState() ||
        indexCompare.getPredicate() != comb::ICmpPredicate::eq)
      return {};
    // `IndexValue` must be same.
    if (indexValue && indexValue != indexCompare.getLhs())
      return {};
    if (!indexValue)
      indexValue = indexCompare.getLhs();
    if (getConstantValue(indexCompare.getRhs()) != idx)
      return {};
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfterValue(reg);
  Value commonConditionValue;
  if (commonConditions.empty())
    commonConditionValue = getOrCreateConstant(reg.getLoc(), APInt(1, 1));
  else
    commonConditionValue = builder.createOrFold<comb::AndOp>(
        reg.getLoc(), builder.getI1Type(), commonConditions.takeVector(), true);
  return std::make_tuple(commonConditionValue, indexValue, trueVal);
}

void FirRegLower::createTree(OpBuilder &builder, Value reg, Value term,
                             Value next) {

  SmallVector<std::tuple<Block *, Value, Value, Value>> worklist;
  auto addToWorklist = [&](Value reg, Value term, Value next) {
    worklist.push_back({builder.getBlock(), reg, term, next});
  };

  auto getArrayIndex = [&](Value reg, Value idx) {
    // Create an array index op just after `reg`.
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfterValue(reg);
    return builder.create<sv::ArrayIndexInOutOp>(reg.getLoc(), reg, idx);
  };

  SmallVector<Value, 8> opsToDelete;
  addToWorklist(reg, term, next);
  while (!worklist.empty()) {
    OpBuilder::InsertionGuard guard(builder);
    Block *block;
    Value reg, term, next;
    std::tie(block, reg, term, next) = worklist.pop_back_val();
    builder.setInsertionPointToEnd(block);
    if (areEquivalentValues(term, next))
      continue;

    auto mux = next.getDefiningOp<comb::MuxOp>();
    if (mux && mux.getTwoState()) {
      addToIfBlock(
          builder, mux.getCond(),
          [&]() { addToWorklist(reg, term, mux.getTrueValue()); },
          [&]() { addToWorklist(reg, term, mux.getFalseValue()); });
      continue;
    }
    // If the next value is an array creation, split the value into
    // invidial elements and construct trees recursively.
    if (auto array = next.getDefiningOp<hw::ArrayCreateOp>()) {
      // First, try restoring subaccess assignments.
      if (auto matchResultOpt =
              tryRestoringSubaccess(builder, reg, term, array)) {
        Value cond, index, trueValue;
        std::tie(cond, index, trueValue) = *matchResultOpt;
        addToIfBlock(
            builder, cond,
            [&]() {
              Value nextReg = getArrayIndex(reg, index);
              // Create a value to use for equivalence checking in the
              // recursive calls. Add the value to `opsToDelete` so that it can
              // be deleted afterwards.
              auto termElement =
                  builder.create<hw::ArrayGetOp>(term.getLoc(), term, index);
              opsToDelete.push_back(termElement);
              addToWorklist(nextReg, termElement, trueValue);
            },
            []() {});
        ++numSubaccessRestored;
        continue;
      }
      // Compat fix for GCC12's libstdc++, cannot use
      // llvm::enumerate(llvm::reverse(OperandRange)).  See #4900.
      // SmallVector<Value> reverseOpValues(llvm::reverse(array.getOperands()));
      for (auto [idx, value] : llvm::enumerate(array.getOperands())) {
        idx = array.getOperands().size() - idx - 1;
        // Create an index constant.
        auto idxVal = getOrCreateConstant(
            array.getLoc(),
            APInt(std::max(1u, llvm::Log2_64_Ceil(array.getOperands().size())),
                  idx));

        auto &index = arrayIndexCache[{reg, idx}];
        if (!index)
          index = getArrayIndex(reg, idxVal);

        // Create a value to use for equivalence checking in the
        // recursive calls. Add the value to `opsToDelete` so that it can
        // be deleted afterwards.
        auto termElement =
            builder.create<hw::ArrayGetOp>(term.getLoc(), term, idxVal);
        opsToDelete.push_back(termElement);
        addToWorklist(index, termElement, value);
      }
      continue;
    }

    builder.create<sv::PAssignOp>(term.getLoc(), reg, next);
  }

  while (!opsToDelete.empty()) {
    auto value = opsToDelete.pop_back_val();
    assert(value.use_empty());
    value.getDefiningOp()->erase();
  }
}

FirRegLower::RegLowerInfo FirRegLower::lower(FirRegOp reg) {
  Location loc = reg.getLoc();

  ImplicitLocOpBuilder builder(reg.getLoc(), reg);
  RegLowerInfo svReg{nullptr, reg.getPresetAttr(), nullptr, nullptr, -1, 0};
  svReg.reg = builder.create<sv::RegOp>(loc, reg.getType(), reg.getNameAttr());
  svReg.width = hw::getBitWidth(reg.getResult().getType());

  if (auto attr = reg->getAttrOfType<IntegerAttr>("firrtl.random_init_start"))
    svReg.randStart = attr.getUInt();

  // Don't move these over
  reg->removeAttr("firrtl.random_init_start");

  // Move Attributes
  svReg.reg->setDialectAttrs(reg->getDialectAttrs());

  if (auto innerSymAttr = reg.getInnerSymAttr())
    svReg.reg.setInnerSymAttr(innerSymAttr);

  auto regVal = builder.create<sv::ReadInOutOp>(loc, svReg.reg);

  if (reg.hasReset()) {
    addToAlwaysBlock(
        module.getBodyBlock(), sv::EventControl::AtPosEdge, reg.getClk(),
        [&](OpBuilder &b) {
          // If this is an AsyncReset, ensure that we emit a self connect to
          // avoid erroneously creating a latch construct.
          if (reg.getIsAsync() && areEquivalentValues(reg, reg.getNext()))
            b.create<sv::PAssignOp>(reg.getLoc(), svReg.reg, reg);
          else
            createTree(b, svReg.reg, reg, reg.getNext());
        },
        reg.getIsAsync() ? ResetType::AsyncReset : ResetType::SyncReset,
        sv::EventControl::AtPosEdge, reg.getReset(),
        [&](OpBuilder &builder) {
          builder.create<sv::PAssignOp>(loc, svReg.reg, reg.getResetValue());
        });
    if (reg.getIsAsync()) {
      svReg.asyncResetSignal = reg.getReset();
      svReg.asyncResetValue = reg.getResetValue();
    }
  } else {
    addToAlwaysBlock(
        module.getBodyBlock(), sv::EventControl::AtPosEdge, reg.getClk(),
        [&](OpBuilder &b) { createTree(b, svReg.reg, reg, reg.getNext()); });
  }

  reg.replaceAllUsesWith(regVal.getResult());
  reg.erase();

  return svReg;
}

// Initialize registers by assigning each element recursively instead of
// initializing entire registers. This is necessary as a workaround for
// verilator which allocates many local variables for concat op.
void FirRegLower::initializeRegisterElements(Location loc, OpBuilder &builder,
                                             Value reg, Value randomSource,
                                             unsigned &pos) {
  auto type = reg.getType().cast<sv::InOutType>().getElementType();
  if (auto intTy = hw::type_dyn_cast<IntegerType>(type)) {
    // Use randomSource[pos-1:pos-width] as a random value.
    pos -= intTy.getWidth();
    auto elem = builder.createOrFold<comb::ExtractOp>(loc, randomSource, pos,
                                                      intTy.getWidth());
    builder.create<sv::BPAssignOp>(loc, reg, elem);
  } else if (auto array = hw::type_dyn_cast<hw::ArrayType>(type)) {
    for (unsigned i = 0, e = array.getSize(); i < e; ++i) {
      auto index = getOrCreateConstant(loc, APInt(llvm::Log2_64_Ceil(e), i));
      initializeRegisterElements(
          loc, builder, builder.create<sv::ArrayIndexInOutOp>(loc, reg, index),
          randomSource, pos);
    }
  } else if (auto structType = hw::type_dyn_cast<hw::StructType>(type)) {
    for (auto e : structType.getElements())
      initializeRegisterElements(
          loc, builder,
          builder.create<sv::StructFieldInOutOp>(loc, reg, e.name),
          randomSource, pos);
  } else {
    assert(false && "unsupported type");
  }
}

void FirRegLower::initialize(OpBuilder &builder, RegLowerInfo reg,
                             ArrayRef<Value> rands) {
  auto loc = reg.reg.getLoc();
  SmallVector<Value> nibbles;
  if (reg.width == 0)
    return;

  uint64_t width = reg.width;
  uint64_t offset = reg.randStart;
  while (width) {
    auto index = offset / 32;
    auto start = offset % 32;
    auto nwidth = std::min(32 - start, width);
    auto elemVal = builder.create<sv::ReadInOutOp>(loc, rands[index]);
    auto elem =
        builder.createOrFold<comb::ExtractOp>(loc, elemVal, start, nwidth);
    nibbles.push_back(elem);
    offset += nwidth;
    width -= nwidth;
  }
  auto concat = builder.createOrFold<comb::ConcatOp>(loc, nibbles);
  unsigned pos = reg.width;
  // Initialize register elements.
  initializeRegisterElements(loc, builder, reg.reg, concat, pos);
}

void FirRegLower::addToAlwaysBlock(Block *block, sv::EventControl clockEdge,
                                   Value clock,
                                   std::function<void(OpBuilder &)> body,
                                   ::ResetType resetStyle,
                                   sv::EventControl resetEdge, Value reset,
                                   std::function<void(OpBuilder &)> resetBody) {
  auto loc = clock.getLoc();
  auto builder = ImplicitLocOpBuilder::atBlockTerminator(loc, block);
  AlwaysKeyType key{builder.getBlock(), clockEdge, clock,
                    resetStyle,         resetEdge, reset};

  sv::AlwaysOp alwaysOp;
  sv::IfOp insideIfOp;
  if (!emitSeparateAlwaysBlocks) {
    std::tie(alwaysOp, insideIfOp) = alwaysBlocks[key];
  }

  if (!alwaysOp) {
    if (reset) {
      assert(resetStyle != ::ResetType::NoReset);
      // Here, we want to create the following structure with sv.always and
      // sv.if. If `reset` is async, we need to add `reset` to a sensitivity
      // list.
      //
      // sv.always @(clockEdge or reset) {
      //   sv.if (reset) {
      //     resetBody
      //   } else {
      //     body
      //   }
      // }

      auto createIfOp = [&]() {
        // It is weird but intended. Here we want to create an empty sv.if
        // with an else block.
        insideIfOp = builder.create<sv::IfOp>(
            reset, []() {}, []() {});
      };
      if (resetStyle == ::ResetType::AsyncReset) {
        sv::EventControl events[] = {clockEdge, resetEdge};
        Value clocks[] = {clock, reset};

        alwaysOp = builder.create<sv::AlwaysOp>(events, clocks, [&]() {
          if (resetEdge == sv::EventControl::AtNegEdge)
            llvm_unreachable("negative edge for reset is not expected");
          createIfOp();
        });
      } else {
        alwaysOp = builder.create<sv::AlwaysOp>(clockEdge, clock, createIfOp);
      }
    } else {
      assert(!resetBody);
      alwaysOp = builder.create<sv::AlwaysOp>(clockEdge, clock);
      insideIfOp = nullptr;
    }
  }

  if (reset) {
    assert(insideIfOp && "reset body must be initialized before");
    auto resetBuilder =
        ImplicitLocOpBuilder::atBlockEnd(loc, insideIfOp.getThenBlock());
    resetBody(resetBuilder);

    auto bodyBuilder =
        ImplicitLocOpBuilder::atBlockEnd(loc, insideIfOp.getElseBlock());
    body(bodyBuilder);
  } else {
    auto bodyBuilder =
        ImplicitLocOpBuilder::atBlockEnd(loc, alwaysOp.getBodyBlock());
    body(bodyBuilder);
  }

  if (!emitSeparateAlwaysBlocks) {
    alwaysBlocks[key] = {alwaysOp, insideIfOp};
  }
}

void SeqToSVPass::runOnOperation() {
  ModuleOp top = getOperation();

  MLIRContext &ctxt = getContext();
  ConversionTarget target(ctxt);
  target.addIllegalDialect<SeqDialect>();
  target.addLegalDialect<sv::SVDialect, comb::CombDialect, hw::HWDialect>();
  RewritePatternSet patterns(&ctxt);
  patterns.add<CompRegLower<CompRegOp>>(&ctxt, lowerToAlwaysFF);
  patterns.add<CompRegLower<CompRegClockEnabledOp>>(&ctxt, lowerToAlwaysFF);
  patterns.add<ClockGateLowering>(&ctxt);

  if (failed(applyPartialConversion(top, target, std::move(patterns))))
    signalPassFailure();
}

void SeqFIRRTLToSVPass::runOnOperation() {
  hw::HWModuleOp module = getOperation();
  FirRegLower firRegLower(module, disableRegRandomization,
                          emitSeparateAlwaysBlocks);
  firRegLower.lower();
  numSubaccessRestored += firRegLower.numSubaccessRestored;
}

void SeqFIRRTLInitToSVPass::runOnOperation() {
  ModuleOp top = getOperation();
  OpBuilder builder(top.getBody(), top.getBody()->begin());
  // FIXME: getOrCreate
  builder.create<sv::MacroDeclOp>(top.getLoc(), "RANDOM", nullptr, nullptr);
}

std::unique_ptr<Pass>
circt::seq::createSeqLowerToSVPass(std::optional<bool> lowerToAlwaysFF) {
  auto pass = std::make_unique<SeqToSVPass>();
  if (lowerToAlwaysFF)
    pass->lowerToAlwaysFF = *lowerToAlwaysFF;
  return pass;
}

std::unique_ptr<Pass> circt::seq::createLowerSeqFIRRTLInitToSV() {
  return std::make_unique<SeqFIRRTLInitToSVPass>();
}

std::unique_ptr<Pass> circt::seq::createSeqFIRRTLLowerToSVPass(
    const LowerSeqFIRRTLToSVOptions &options) {
  return std::make_unique<SeqFIRRTLToSVPass>(options);
}

//===- PrepareForEmission.cpp - IR Prepass for Emitter --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the "prepare" pass that walks the IR before the emitter
// gets involved.  This allows us to do some transformations that would be
// awkward to implement inline in the emitter.
//
// NOTE: This file covers the preparation phase of `ExportVerilog` which mainly
// legalizes the IR and makes adjustments necessary for emission. This is the
// place to mutate the IR if emission needs it. The IR cannot be modified during
// emission itself, which happens in parallel.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "ExportVerilogInternals.h"
#include "circt/Conversion/ExportVerilog.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/LTL/LTLDialect.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "prepare-for-emission"

using namespace circt;
using namespace comb;
using namespace hw;
using namespace sv;
using namespace ExportVerilog;

// Check if the value is from read of a wire or reg or is a port.
bool ExportVerilog::isSimpleReadOrPort(Value v) {
  if (v.isa<BlockArgument>())
    return true;
  auto vOp = v.getDefiningOp();
  if (!vOp)
    return false;
  auto read = dyn_cast<ReadInOutOp>(vOp);
  if (!read)
    return false;
  auto readSrc = read.getInput().getDefiningOp();
  if (!readSrc)
    return false;
  return isa<sv::WireOp, RegOp, LogicOp, XMROp, XMRRefOp>(readSrc);
}

// Check if the value is deemed worth spilling into a wire.
static bool shouldSpillWire(Operation &op, const LoweringOptions &options) {
  if (!isVerilogExpression(&op))
    return false;

  // Spill temporary wires if it is not possible to inline.
  return !ExportVerilog::isExpressionEmittedInline(&op, options);
}

// Given an instance, make sure all inputs are driven from wires or ports.
static void spillWiresForInstanceInputs(InstanceOp op) {
  Block *block = op->getParentOfType<HWModuleOp>().getBodyBlock();
  auto builder = ImplicitLocOpBuilder::atBlockBegin(op.getLoc(), block);

  SmallString<32> nameTmp{"_", op.getInstanceName(), "_"};
  auto namePrefixSize = nameTmp.size();

  size_t nextOpNo = 0;
  auto ports = op.getPortList();
  for (auto &port : ports.getInputs()) {
    auto src = op.getOperand(nextOpNo);
    ++nextOpNo;

    if (isSimpleReadOrPort(src))
      continue;

    nameTmp.resize(namePrefixSize);
    if (port.name)
      nameTmp += port.name.getValue().str();
    else
      nameTmp += std::to_string(nextOpNo - 1);

    auto newWire = builder.create<sv::WireOp>(src.getType(), nameTmp);
    auto newWireRead = builder.create<ReadInOutOp>(newWire);
    auto connect = builder.create<AssignOp>(newWire, src);
    newWireRead->moveBefore(op);
    connect->moveBefore(op);
    op.setOperand(nextOpNo - 1, newWireRead);
  }
}

// Ensure that each output of an instance are used only by a wire
static void lowerInstanceResults(InstanceOp op) {
  Block *block = op->getParentOfType<HWModuleOp>().getBodyBlock();
  auto builder = ImplicitLocOpBuilder::atBlockBegin(op.getLoc(), block);

  SmallString<32> nameTmp{"_", op.getInstanceName(), "_"};
  auto namePrefixSize = nameTmp.size();

  size_t nextResultNo = 0;
  auto ports = op.getPortList();
  for (auto &port : ports.getOutputs()) {
    auto result = op.getResult(nextResultNo);
    ++nextResultNo;

    // If the result doesn't have a user, the connection won't be emitted by
    // Emitter, so there's no need to create a wire for it. However, if the
    // result is a zero bit value, the normal emission code path should be used,
    // as zero bit values require special handling by the emitter.
    if (result.use_empty() && !ExportVerilog::isZeroBitType(result.getType()))
      continue;

    if (result.hasOneUse()) {
      OpOperand &use = *result.getUses().begin();
      if (dyn_cast_or_null<OutputOp>(use.getOwner()))
        continue;
      if (auto assign = dyn_cast_or_null<AssignOp>(use.getOwner())) {
        // Move assign op after instance to resolve cyclic dependencies.
        assign->moveAfter(op);
        continue;
      }
    }

    nameTmp.resize(namePrefixSize);
    if (port.name)
      nameTmp += port.name.getValue().str();
    else
      nameTmp += std::to_string(nextResultNo - 1);
    Value newWire = builder.create<sv::WireOp>(result.getType(), nameTmp);

    while (!result.use_empty()) {
      auto newWireRead = builder.create<ReadInOutOp>(newWire);
      OpOperand &use = *result.getUses().begin();
      use.set(newWireRead);
      newWireRead->moveBefore(use.getOwner());
    }

    auto connect = builder.create<AssignOp>(newWire, result);
    connect->moveAfter(op);
  }
}

/// Emit an explicit wire or logic to assign operation's result. This function
/// is used to create a temporary to legalize a verilog expression or to
/// resolve use-before-def in a graph region. If `emitWireAtBlockBegin` is true,
/// a temporary wire will be created at the beginning of the block. Otherwise,
/// a wire is created just after op's position so that we can inline the
/// assignment into its wire declaration.
static void lowerUsersToTemporaryWire(Operation &op,
                                      bool emitWireAtBlockBegin = false) {
  Block *block = op.getBlock();
  auto builder = ImplicitLocOpBuilder::atBlockBegin(op.getLoc(), block);
  bool isProceduralRegion = op.getParentOp()->hasTrait<ProceduralRegion>();

  auto createWireForResult = [&](Value result, StringAttr name) {
    Value newWire;
    // If the op is in a procedural region, use logic op.
    if (isProceduralRegion)
      newWire = builder.create<LogicOp>(result.getType(), name);
    else
      newWire = builder.create<sv::WireOp>(result.getType(), name);

    while (!result.use_empty()) {
      auto newWireRead = builder.create<ReadInOutOp>(newWire);
      OpOperand &use = *result.getUses().begin();
      use.set(newWireRead);
      newWireRead->moveBefore(use.getOwner());
    }

    Operation *connect;
    if (isProceduralRegion)
      connect = builder.create<BPAssignOp>(newWire, result);
    else
      connect = builder.create<AssignOp>(newWire, result);
    connect->moveAfter(&op);

    // Move the temporary to the appropriate place.
    if (!emitWireAtBlockBegin) {
      // `emitWireAtBlockBegin` is intendend to be used for resovling cyclic
      // dependencies. So when `emitWireAtBlockBegin` is true, we keep the
      // position of the wire. Otherwise, we move the wire to immediately after
      // the expression so that the wire and assignment are next to each other.
      // This ordering will be used by the heurstic to inline assignments.
      newWire.getDefiningOp()->moveAfter(&op);
    }
  };

  // If the op has a single result, infer a meaningful name from the
  // value.
  if (op.getNumResults() == 1) {
    auto namehint = inferStructuralNameForTemporary(op.getResult(0));
    op.removeAttr("sv.namehint");
    createWireForResult(op.getResult(0), namehint);
    return;
  }

  // If the op has multiple results, create wires for each result.
  for (auto result : op.getResults())
    createWireForResult(result, StringAttr());
}

// Given a side effect free "always inline" operation, make sure that it
// exists in the same block as its users and that it has one use for each one.
static void lowerAlwaysInlineOperation(Operation *op,
                                       const LoweringOptions &options) {
  assert(op->getNumResults() == 1 &&
         "only support 'always inline' ops with one result");

  // Moving/cloning an op should pull along its operand tree with it if they are
  // always inline.  This happens when an array index has a constant operand for
  // example.  If the operand is not always inline, then evaluate it to see if
  // it should be spilled to a wire.
  auto recursivelyHandleOperands = [&](Operation *op) {
    for (auto operand : op->getOperands()) {
      if (auto *operandOp = operand.getDefiningOp()) {
        if (isExpressionAlwaysInline(operandOp))
          lowerAlwaysInlineOperation(operandOp, options);
        else if (shouldSpillWire(*operandOp, options))
          lowerUsersToTemporaryWire(*operandOp);
      }
    }
  };

  // If an operation is an assignment that immediately follows the declaration
  // of its wire, return that wire. Otherwise return the original op. This
  // ensures that the declaration and assignment don't get split apart by
  // inlined operations, which allows `ExportVerilog` to trivially emit the
  // expression inline in the declaration.
  auto skipToWireImmediatelyBefore = [](Operation *user) {
    if (!isa<BPAssignOp, AssignOp>(user))
      return user;
    auto *wireOp = user->getOperand(0).getDefiningOp();
    if (wireOp && wireOp->getNextNode() == user)
      return wireOp;
    return user;
  };

  // If this operation has multiple uses, duplicate it into N-1 of them in
  // turn.
  while (!op->hasOneUse()) {
    OpOperand &use = *op->getUses().begin();
    Operation *user = skipToWireImmediatelyBefore(use.getOwner());

    // Clone the op before the user.
    auto *newOp = op->clone();
    user->getBlock()->getOperations().insert(Block::iterator(user), newOp);
    // Change the user to use the new op.
    use.set(newOp->getResult(0));

    // If any of the operations of the moved op are always inline, recursively
    // handle them too.
    recursivelyHandleOperands(newOp);
  }

  // Finally, ensures the op is in the same block as its user so it can be
  // inlined.
  Operation *user = skipToWireImmediatelyBefore(*op->getUsers().begin());
  op->moveBefore(user);

  // If any of the operations of the moved op are always inline, recursively
  // move/clone them too.
  recursivelyHandleOperands(op);
  return;
}

// Find a nearest insertion point where logic op can be declared.
// Logic ops are emitted as "automatic logic" in procedural regions, but
// they must be declared at beginning of blocks.
static std::pair<Block *, Block::iterator>
findLogicOpInsertionPoint(Operation *op) {
  // We have to skip `ifdef.procedural` because it is a just macro.
  if (isa<IfDefProceduralOp>(op->getParentOp()))
    return findLogicOpInsertionPoint(op->getParentOp());
  return {op->getBlock(), op->getBlock()->begin()};
}

/// Lower a variadic fully-associative operation into an expression tree.  This
/// enables long-line splitting to work with them.
/// NOLINTNEXTLINE(misc-no-recursion)
static Value lowerFullyAssociativeOp(Operation &op, OperandRange operands,
                                     SmallVector<Operation *> &newOps) {
  // save the top level name
  auto name = op.getAttr("sv.namehint");
  if (name)
    op.removeAttr("sv.namehint");
  Value lhs, rhs;
  switch (operands.size()) {
  case 0:
    assert(0 && "cannot be called with empty operand range");
    break;
  case 1:
    return operands[0];
  case 2:
    lhs = operands[0];
    rhs = operands[1];
    break;
  default:
    auto firstHalf = operands.size() / 2;
    lhs = lowerFullyAssociativeOp(op, operands.take_front(firstHalf), newOps);
    rhs = lowerFullyAssociativeOp(op, operands.drop_front(firstHalf), newOps);
    break;
  }

  OperationState state(op.getLoc(), op.getName());
  state.addOperands(ValueRange{lhs, rhs});
  state.addTypes(op.getResult(0).getType());
  auto *newOp = Operation::create(state);
  op.getBlock()->getOperations().insert(Block::iterator(&op), newOp);
  newOps.push_back(newOp);
  if (name)
    newOp->setAttr("sv.namehint", name);
  if (auto twoState = op.getAttr("twoState"))
    newOp->setAttr("twoState", twoState);
  return newOp->getResult(0);
}

/// Transform "a + -cst" ==> "a - cst" for prettier output.  This returns the
/// first operation emitted.
static Operation *rewriteAddWithNegativeConstant(comb::AddOp add,
                                                 hw::ConstantOp rhsCst) {
  ImplicitLocOpBuilder builder(add.getLoc(), add);

  // Get the positive constant.
  auto negCst = builder.create<hw::ConstantOp>(-rhsCst.getValue());
  auto sub =
      builder.create<comb::SubOp>(add.getOperand(0), negCst, add.getTwoState());
  add.getResult().replaceAllUsesWith(sub);
  add.erase();
  if (rhsCst.use_empty())
    rhsCst.erase();
  return negCst;
}

// Transforms a hw.struct_explode operation into a set of hw.struct_extract
// operations, and returns the first op generated.
static Operation *lowerStructExplodeOp(hw::StructExplodeOp op) {
  Operation *firstOp = nullptr;
  ImplicitLocOpBuilder builder(op.getLoc(), op);
  StructType structType = op.getInput().getType().cast<StructType>();
  for (auto [res, field] :
       llvm::zip(op.getResults(), structType.getElements())) {
    auto extract =
        builder.create<hw::StructExtractOp>(op.getInput(), field.name);
    if (!firstOp)
      firstOp = extract;
    res.replaceAllUsesWith(extract);
  }
  op.erase();
  return firstOp;
}

/// Given an operation in a procedural region, scan up the region tree to find
/// the first operation in a graph region (typically an always or initial op).
///
/// By looking for a graph region, we will stop at graph-region #ifdef's that
/// may enclose this operation.
static Operation *findParentInNonProceduralRegion(Operation *op) {
  Operation *parentOp = op->getParentOp();
  assert(parentOp->hasTrait<ProceduralRegion>() &&
         "we should only be hoisting from procedural");
  while (parentOp->getParentOp()->hasTrait<ProceduralRegion>())
    parentOp = parentOp->getParentOp();
  return parentOp;
}

/// This function is invoked on side effecting Verilog expressions when we're in
/// 'disallowLocalVariables' mode for old Verilog clients.  This ensures that
/// any side effecting expressions are only used by a single BPAssign to a
/// sv.reg or sv.logic operation.  This ensures that the verilog emitter doesn't
/// have to worry about spilling them.
///
/// This returns true if the op was rewritten, false otherwise.
static bool rewriteSideEffectingExpr(Operation *op) {
  assert(op->getNumResults() == 1 && "isn't a verilog expression");

  // Check to see if this is already rewritten.
  if (op->hasOneUse()) {
    if (auto assign = dyn_cast<BPAssignOp>(*op->user_begin()))
      return false;
  }

  // Otherwise, we have to transform it.  Insert a reg at the top level, make
  // everything using the side effecting expression read the reg, then assign to
  // it after the side effecting operation.
  Value opValue = op->getResult(0);

  // Scan to the top of the region tree to find out where to insert the reg.
  Operation *parentOp = findParentInNonProceduralRegion(op);
  OpBuilder builder(parentOp);
  auto reg = builder.create<RegOp>(op->getLoc(), opValue.getType());

  // Everything using the expr now uses a read_inout of the reg.
  auto value = builder.create<ReadInOutOp>(op->getLoc(), reg);
  opValue.replaceAllUsesWith(value);

  // We assign the side effect expr to the reg immediately after that expression
  // is computed.
  builder.setInsertionPointAfter(op);
  builder.create<BPAssignOp>(op->getLoc(), reg, opValue);
  return true;
}

/// This function is called for non-side-effecting Verilog expressions when
/// we're in 'disallowLocalVariables' mode for old Verilog clients.  It hoists
/// non-constant expressions out to the top level so they don't turn into local
/// variable declarations.
static bool hoistNonSideEffectExpr(Operation *op) {
  // Never hoist "always inline" expressions except for inout stuffs - they will
  // never generate a temporary and in fact must always be emitted inline.
  if (isExpressionAlwaysInline(op) &&
      !(isa<sv::ReadInOutOp>(op) ||
        op->getResult(0).getType().isa<hw::InOutType>()))
    return false;

  // Scan to the top of the region tree to find out where to move the op.
  Operation *parentOp = findParentInNonProceduralRegion(op);

  // We can typically hoist all the way out to the top level in one step, but
  // there may be intermediate operands that aren't hoistable.  If so, just
  // hoist one level.
  bool cantHoist = false;
  if (llvm::any_of(op->getOperands(), [&](Value operand) -> bool {
        // The operand value dominates the original operation, but may be
        // defined in one of the procedural regions between the operation and
        // the top level of the module.  We can tell this quite efficiently by
        // looking for ops in a procedural region - because procedural regions
        // live in graph regions but not visa-versa.
        if (BlockArgument block = operand.dyn_cast<BlockArgument>()) {
          // References to ports are always ok.
          if (isa<hw::HWModuleOp>(block.getParentBlock()->getParentOp()))
            return false;

          cantHoist = true;
          return true;
        }
        Operation *operandOp = operand.getDefiningOp();

        if (operandOp->getParentOp()->hasTrait<ProceduralRegion>()) {
          cantHoist |= operandOp->getBlock() == op->getBlock();
          return true;
        }
        return false;
      })) {

    // If the operand is in the same block as the expression then we can't hoist
    // this out at all.
    if (cantHoist)
      return false;

    // Otherwise, we can hoist it, but not all the way out in one step.  Just
    // hoist one level out.
    parentOp = op->getParentOp();
  }

  op->moveBefore(parentOp);
  return true;
}

/// Check whether an op is a declaration that can be moved.
static bool isMovableDeclaration(Operation *op) {
  return op->getNumResults() == 1 &&
         op->getResult(0).getType().isa<InOutType, sv::InterfaceType>() &&
         op->getNumOperands() == 0;
}

//===----------------------------------------------------------------------===//
// EmittedExpressionStateManager
//===----------------------------------------------------------------------===//

struct EmittedExpressionState {
  size_t size = 0;
  static EmittedExpressionState getBaseState() {
    return EmittedExpressionState{1};
  }
  void mergeState(const EmittedExpressionState &state) { size += state.size; }
};

/// This class handles information about AST structures of each expressions.
class EmittedExpressionStateManager
    : public hw::TypeOpVisitor<EmittedExpressionStateManager,
                               EmittedExpressionState>,
      public comb::CombinationalVisitor<EmittedExpressionStateManager,
                                        EmittedExpressionState>,
      public sv::Visitor<EmittedExpressionStateManager,
                         EmittedExpressionState> {
public:
  EmittedExpressionStateManager(const LoweringOptions &options)
      : options(options){};

  // Get or caluculate an emitted expression state.
  EmittedExpressionState getExpressionState(Value value);

  bool dispatchHeuristic(Operation &op);

  // Return true if the operation is worth spilling based on the expression
  // state of the op.
  bool shouldSpillWireBasedOnState(Operation &op);

private:
  friend class TypeOpVisitor<EmittedExpressionStateManager,
                             EmittedExpressionState>;
  friend class CombinationalVisitor<EmittedExpressionStateManager,
                                    EmittedExpressionState>;
  friend class Visitor<EmittedExpressionStateManager, EmittedExpressionState>;
  EmittedExpressionState visitUnhandledExpr(Operation *op) {
    LLVM_DEBUG(op->emitWarning() << "unhandled by EmittedExpressionState");
    if (op->getNumOperands() == 0)
      return EmittedExpressionState::getBaseState();
    return mergeOperandsStates(op);
  }
  EmittedExpressionState visitInvalidComb(Operation *op) {
    return dispatchTypeOpVisitor(op);
  }
  EmittedExpressionState visitUnhandledComb(Operation *op) {
    return visitUnhandledExpr(op);
  }
  EmittedExpressionState visitInvalidTypeOp(Operation *op) {
    return dispatchSVVisitor(op);
  }
  EmittedExpressionState visitUnhandledTypeOp(Operation *op) {
    return visitUnhandledExpr(op);
  }
  EmittedExpressionState visitUnhandledSV(Operation *op) {
    return visitUnhandledExpr(op);
  }

  using CombinationalVisitor::visitComb;
  using Visitor::visitSV;

  const LoweringOptions &options;

  // A helper function to accumulate states.
  EmittedExpressionState mergeOperandsStates(Operation *op);

  // This caches the expression states in the module scope.
  DenseMap<Value, EmittedExpressionState> expressionStates;
};

EmittedExpressionState
EmittedExpressionStateManager::getExpressionState(Value v) {
  auto it = expressionStates.find(v);
  if (it != expressionStates.end())
    return it->second;

  // Ports.
  if (auto blockArg = v.dyn_cast<BlockArgument>())
    return EmittedExpressionState::getBaseState();

  EmittedExpressionState state =
      dispatchCombinationalVisitor(v.getDefiningOp());
  expressionStates.insert({v, state});
  return state;
}

EmittedExpressionState
EmittedExpressionStateManager::mergeOperandsStates(Operation *op) {
  EmittedExpressionState state;
  for (auto operand : op->getOperands())
    state.mergeState(getExpressionState(operand));

  return state;
}

/// If exactly one use of this op is an assign, replace the other uses with a
/// read from the assigned wire or reg. This assumes the preconditions for doing
/// so are met: op must be an expression in a non-procedural region.
static bool reuseExistingInOut(Operation *op, const LoweringOptions &options) {
  // Try to collect a single assign and all the other uses of op.
  sv::AssignOp assign;
  SmallVector<OpOperand *> uses;

  // Look at each use.
  for (OpOperand &use : op->getUses()) {
    // If it's an assign, try to save it.
    if (auto assignUse = dyn_cast<AssignOp>(use.getOwner())) {
      // If there are multiple assigns, bail out.
      if (assign)
        return false;

      // If the assign is not at the top level, it might be conditionally
      // executed. So bail out.
      if (!isa<HWModuleOp>(assignUse->getParentOp()))
        return false;

      // Remember this assign for later.
      assign = assignUse;
      continue;
    }

    // If not an assign, remember this use for later.
    uses.push_back(&use);
  }

  // If we didn't find anything, bail out.
  if (!assign || uses.empty())
    return false;

  if (auto *cop = assign.getSrc().getDefiningOp())
    if (isa<ConstantOp>(cop))
      return false;

  // Replace all saved uses with a read from the assigned destination.
  ImplicitLocOpBuilder builder(assign.getDest().getLoc(), op->getContext());
  for (OpOperand *use : uses) {
    builder.setInsertionPoint(use->getOwner());
    auto read = builder.create<ReadInOutOp>(assign.getDest());
    use->set(read);
  }
  if (auto *destOp = assign.getDest().getDefiningOp())
    if (isExpressionAlwaysInline(destOp))
      lowerAlwaysInlineOperation(destOp, options);
  return true;
}

bool EmittedExpressionStateManager::dispatchHeuristic(Operation &op) {
  // TODO: Consider using virtual functions.
  if (options.isWireSpillingHeuristicEnabled(
          LoweringOptions::SpillLargeTermsWithNamehints))
    if (auto hint = op.getAttrOfType<StringAttr>("sv.namehint")) {
      // Spill wires if the name doesn't have a prefix "_".
      if (!hint.getValue().startswith("_"))
        return true;
      // If the name has prefix "_", spill if the size is greater than the
      // threshould.
      if (getExpressionState(op.getResult(0)).size >=
          options.wireSpillingNamehintTermLimit)
        return true;
    }

  return false;
}

/// Return true if it is beneficial to spill the operation under the specified
/// spilling heuristic.
bool EmittedExpressionStateManager::shouldSpillWireBasedOnState(Operation &op) {
  // Don't spill wires for inout operations and simple expressions such as read
  // or constant.
  if (op.getNumResults() == 0 ||
      op.getResult(0).getType().isa<hw::InOutType>() ||
      isa<ReadInOutOp, ConstantOp>(op))
    return false;

  // If the operation is only used by an assignment, the op is already spilled
  // to a wire.
  if (op.hasOneUse()) {
    auto *singleUser = *op.getUsers().begin();
    if (isa<hw::OutputOp, sv::AssignOp, sv::BPAssignOp, hw::InstanceOp>(
            singleUser))
      return false;

    // If the single user is bitcast, we check the same property for the bitcast
    // op since bitcast op is no-op in system verilog.
    if (singleUser->hasOneUse() && isa<hw::BitcastOp>(singleUser) &&
        isa<hw::OutputOp, sv::AssignOp, sv::BPAssignOp>(
            *singleUser->getUsers().begin()))
      return false;
  }

  // If the term size is greater than `maximumNumberOfTermsPerExpression`,
  // we have to spill the wire.
  if (options.maximumNumberOfTermsPerExpression <
      getExpressionState(op.getResult(0)).size)
    return true;
  return dispatchHeuristic(op);
}

/// After the legalization, we are able to know accurate verilog AST structures.
/// So this function walks and prettifies verilog IR with a heuristic method
/// specified by `options.wireSpillingHeuristic` based on the structures.
static void prettifyAfterLegalization(
    Block &block, EmittedExpressionStateManager &expressionStateManager) {
  // TODO: Handle procedural regions as well.
  if (block.getParentOp()->hasTrait<ProceduralRegion>())
    return;

  for (auto &op : llvm::make_early_inc_range(block)) {
    if (!isVerilogExpression(&op))
      continue;
    if (expressionStateManager.shouldSpillWireBasedOnState(op)) {
      lowerUsersToTemporaryWire(op);
      continue;
    }
  }

  for (auto &op : block) {
    // If the operations has regions, visit each of the region bodies.
    for (auto &region : op.getRegions()) {
      if (!region.empty())
        prettifyAfterLegalization(region.front(), expressionStateManager);
    }
  }
}

struct WireLowering {
  hw::WireOp wireOp;
  Block::iterator declarePoint;
  Block::iterator assignPoint;
};

/// Determine the insertion location of declaration and assignment ops for
/// `hw::WireOp`s in a block. This function tries to place the declaration point
/// as late as possible, right before the very first use of the wire. It also
/// tries to place the assign point as early as possible, right after the
/// assigned value becomes available.
static void buildWireLowerings(Block &block,
                               SmallVectorImpl<WireLowering> &wireLowerings) {
  for (auto hwWireOp : block.getOps<hw::WireOp>()) {
    // Find the earliest point in the block at which the input operand is
    // available. The assign operation will have to go after that to not create
    // a use-before-def situation.
    Block::iterator assignPoint = block.begin();
    if (auto *defOp = hwWireOp.getInput().getDefiningOp())
      if (defOp && defOp->getBlock() == &block)
        assignPoint = ++Block::iterator(defOp);

    // Find the earliest use of the wire within the block. The wire declaration
    // will have to go somewhere before this point to not create a
    // use-before-def situation.
    Block::iterator declarePoint = assignPoint;
    for (auto *user : hwWireOp->getUsers()) {
      while (user->getBlock() != &block)
        user = user->getParentOp();
      if (user->isBeforeInBlock(&*declarePoint))
        declarePoint = Block::iterator(user);
    }

    wireLowerings.push_back({hwWireOp, declarePoint, assignPoint});
  }
}

/// Materialize the SV wire declaration and assignment operations in the
/// locations previously determined by a call to `buildWireLowerings`. This
/// replaces all `hw::WireOp`s with their appropriate SV counterpart.
static void applyWireLowerings(Block &block,
                               ArrayRef<WireLowering> wireLowerings) {
  bool isProceduralRegion = block.getParentOp()->hasTrait<ProceduralRegion>();

  for (auto [hwWireOp, declarePoint, assignPoint] : wireLowerings) {
    // Create the declaration.
    ImplicitLocOpBuilder builder(hwWireOp.getLoc(), &block, declarePoint);
    Value decl;
    if (isProceduralRegion) {
      decl = builder.create<LogicOp>(hwWireOp.getType(), hwWireOp.getNameAttr(),
                                     hwWireOp.getInnerSymAttr());
    } else {
      decl =
          builder.create<sv::WireOp>(hwWireOp.getType(), hwWireOp.getNameAttr(),
                                     hwWireOp.getInnerSymAttr());
    }

    // Carry attributes over to the lowered operation.
    auto defaultAttrNames = hwWireOp.getAttributeNames();
    for (auto namedAttr : hwWireOp->getAttrs())
      if (!llvm::is_contained(defaultAttrNames, namedAttr.getName()))
        decl.getDefiningOp()->setAttr(namedAttr.getName(),
                                      namedAttr.getValue());

    // Create the assignment. If it is supposed to be at a different point than
    // the declaration, reposition the builder there. Otherwise we'll just build
    // the assignment immediately after the declaration.
    if (assignPoint != declarePoint)
      builder.setInsertionPoint(&block, assignPoint);
    if (isProceduralRegion)
      builder.create<BPAssignOp>(decl, hwWireOp.getInput());
    else
      builder.create<AssignOp>(decl, hwWireOp.getInput());

    // Create the read. If we have created the assignment at a different point
    // than the declaration, reposition the builder to immediately after the
    // declaration. Otherwise we'll just build the read immediately after the
    // assignment.
    if (assignPoint != declarePoint)
      builder.setInsertionPointAfterValue(decl);
    auto readOp = builder.create<sv::ReadInOutOp>(decl);

    // Replace the HW wire.
    hwWireOp.replaceAllUsesWith(readOp.getResult());
  }

  for (auto [hwWireOp, declarePoint, assignPoint] : wireLowerings)
    hwWireOp.erase();
}

/// For each module we emit, do a prepass over the structure, pre-lowering and
/// otherwise rewriting operations we don't want to emit.
static LogicalResult legalizeHWModule(Block &block,
                                      const LoweringOptions &options) {

  // First step, check any nested blocks that exist in this region.  This walk
  // can pull things out to our level of the hierarchy.
  for (auto &op : block) {
    // If the operations has regions, prepare each of the region bodies.
    for (auto &region : op.getRegions()) {
      if (!region.empty())
        if (failed(legalizeHWModule(region.front(), options)))
          return failure();
    }
  }

  // Lower HW wires into SV wires in the appropriate spots. Try to put the
  // declaration close to the first use in the block, and the assignment right
  // after the assigned value becomes available.
  {
    SmallVector<WireLowering> wireLowerings;
    buildWireLowerings(block, wireLowerings);
    applyWireLowerings(block, wireLowerings);
  }

  // Next, walk all of the operations at this level.

  // True if these operations are in a procedural region.
  bool isProceduralRegion = block.getParentOp()->hasTrait<ProceduralRegion>();

  // This tracks "always inline" operation already visited in the iterations to
  // avoid processing same operations infinitely.
  DenseSet<Operation *> visitedAlwaysInlineOperations;

  for (Block::iterator opIterator = block.begin(), e = block.end();
       opIterator != e;) {
    auto &op = *opIterator++;

    if (!isa<CombDialect, SVDialect, HWDialect, ltl::LTLDialect,
             verif::VerifDialect>(op.getDialect())) {
      auto d = op.emitError() << "dialect \"" << op.getDialect()->getNamespace()
                              << "\" not supported for direct Verilog emission";
      d.attachNote() << "ExportVerilog cannot emit this operation; it needs "
                        "to be lowered before running ExportVerilog";
      return failure();
    }

    // Do not reorder LTL expressions, which are always emitted inline.
    if (isa<ltl::LTLDialect>(op.getDialect()))
      continue;

    // Name legalization should have happened in a different pass for these sv
    // elements and we don't want to change their name through re-legalization
    // (e.g. letting a temporary take the name of an unvisited wire). Adding
    // them now ensures any temporary generated will not use one of the names
    // previously declared.
    if (auto instance = dyn_cast<InstanceOp>(op)) {
      // Anchor return values to wires early
      lowerInstanceResults(instance);
      // Anchor ports of instances when `disallowExpressionInliningInPorts` is
      // enabled.
      if (options.disallowExpressionInliningInPorts)
        spillWiresForInstanceInputs(instance);
    }

    // If logic op is located in a procedural region, we have to move the logic
    // op declaration to a valid program point.
    if (isProceduralRegion && isa<LogicOp>(op)) {
      if (options.disallowLocalVariables) {
        // When `disallowLocalVariables` is enabled, "automatic logic" is
        // prohibited so hoist the op to a non-procedural region.
        auto *parentOp = findParentInNonProceduralRegion(&op);
        op.moveBefore(parentOp);
      }
    }

    // If the target doesn't support local variables, hoist all the expressions
    // out to the nearest non-procedural region.
    if (options.disallowLocalVariables && isVerilogExpression(&op) &&
        isProceduralRegion) {

      // Force any side-effecting expressions in nested regions into a sv.reg
      // if we aren't allowing local variable declarations.  The Verilog emitter
      // doesn't want to have to have to know how to synthesize a reg in the
      // case they have to be spilled for whatever reason.
      if (!mlir::isMemoryEffectFree(&op)) {
        if (rewriteSideEffectingExpr(&op))
          continue;
      }

      // Hoist other expressions out to the parent region.
      //
      // NOTE: This effectively disables inlining of expressions into if
      // conditions, $fwrite statements, and instance inputs.  We could be
      // smarter in ExportVerilog itself, but we'd have to teach it to put
      // spilled expressions (due to line length, multiple-uses, and
      // non-inlinable expressions) in the outer scope.
      if (hoistNonSideEffectExpr(&op))
        continue;
    }

    // Duplicate "always inline" expression for each of their users and move
    // them to be next to their users.
    if (isExpressionAlwaysInline(&op)) {
      // Nuke use-less operations.
      if (op.use_empty()) {
        op.erase();
        continue;
      }
      // Process the op only when the op is never processed from the top-level
      // loop.
      if (visitedAlwaysInlineOperations.insert(&op).second)
        lowerAlwaysInlineOperation(&op, options);

      continue;
    }

    if (auto structCreateOp = dyn_cast<hw::StructCreateOp>(op);
        options.disallowPackedStructAssignments && structCreateOp) {
      // Force packed struct assignment to be a wire + assignments to each
      // field.
      Value wireOp;
      ImplicitLocOpBuilder builder(op.getLoc(), &op);
      hw::StructType structType =
          structCreateOp.getResult().getType().cast<hw::StructType>();
      bool procedural = op.getParentOp()->hasTrait<ProceduralRegion>();
      if (procedural)
        wireOp = builder.create<LogicOp>(structType);
      else
        wireOp = builder.create<sv::WireOp>(structType);

      for (auto [input, field] :
           llvm::zip(structCreateOp.getInput(), structType.getElements())) {
        auto target =
            builder.create<sv::StructFieldInOutOp>(wireOp, field.name);
        if (procedural)
          builder.create<BPAssignOp>(target, input);
        else
          builder.create<AssignOp>(target, input);
      }
      // Have to create a separate read for each use to keep things legal.
      for (auto &use :
           llvm::make_early_inc_range(structCreateOp.getResult().getUses()))
        use.set(builder.create<ReadInOutOp>(wireOp));

      structCreateOp.erase();
      continue;
    }

    // Force array index expressions to be a simple name when the
    // `mitigateVivadoArrayIndexConstPropBug` option is set.
    if (options.mitigateVivadoArrayIndexConstPropBug &&
        isa<ArraySliceOp, ArrayGetOp, ArrayIndexInOutOp,
            IndexedPartSelectInOutOp, IndexedPartSelectOp>(&op)) {

      // Check if the index expression is already a wire.
      Value wireOp;
      Value readOp;
      if (auto maybeReadOp =
              op.getOperand(1).getDefiningOp<sv::ReadInOutOp>()) {
        if (isa_and_nonnull<sv::WireOp, LogicOp>(
                maybeReadOp.getInput().getDefiningOp())) {
          wireOp = maybeReadOp.getInput();
          readOp = maybeReadOp;
        }
      }

      // Insert a wire and read if necessary.
      ImplicitLocOpBuilder builder(op.getLoc(), &op);
      if (!wireOp) {
        auto type = op.getOperand(1).getType();
        const auto *name = "_GEN_ARRAY_IDX";
        if (op.getParentOp()->hasTrait<ProceduralRegion>()) {
          wireOp = builder.create<LogicOp>(type, name);
          builder.create<BPAssignOp>(wireOp, op.getOperand(1));
        } else {
          wireOp = builder.create<sv::WireOp>(type, name);
          builder.create<AssignOp>(wireOp, op.getOperand(1));
        }
        readOp = builder.create<ReadInOutOp>(wireOp);
      }
      op.setOperand(1, readOp);

      // Add a `(* keep = "true" *)` SV attribute to the wire.
      sv::addSVAttributes(wireOp.getDefiningOp(),
                          SVAttributeAttr::get(wireOp.getContext(), "keep",
                                               R"("true")",
                                               /*emitAsComment=*/false));
      continue;
    }

    // If this expression is deemed worth spilling into a wire, do it here.
    if (shouldSpillWire(op, options)) {
      // We first check that it is possible to reuse existing wires as a spilled
      // wire. Otherwise, create a new wire op.
      if (isProceduralRegion || !reuseExistingInOut(&op, options)) {
        if (options.disallowLocalVariables) {
          // If we're not in a procedural region, or we are, but we can hoist
          // out of it, we are good to generate a wire.
          if (!isProceduralRegion ||
              (isProceduralRegion && hoistNonSideEffectExpr(&op))) {
            // If op is moved to a non-procedural region, create a temporary
            // wire.
            if (!op.getParentOp()->hasTrait<ProceduralRegion>())
              lowerUsersToTemporaryWire(op);

            // If we're in a procedural region, we move on to the next op in the
            // block. The expression splitting and canonicalization below will
            // happen after we recurse back up. If we're not in a procedural
            // region, the expression can continue being worked on.
            if (isProceduralRegion)
              continue;
          }
        } else {
          // If `disallowLocalVariables` is not enabled, we can spill the
          // expression to automatic logic declarations even when the op is in a
          // procedural region.
          lowerUsersToTemporaryWire(op);
        }
      }
    }

    // Lower variadic fully-associative operations with more than two operands
    // into balanced operand trees so we can split long lines across multiple
    // statements.
    // TODO: This is checking the Commutative property, which doesn't seem
    // right in general.  MLIR doesn't have a "fully associative" property.
    if (op.getNumOperands() > 2 && op.getNumResults() == 1 &&
        op.hasTrait<mlir::OpTrait::IsCommutative>() &&
        mlir::isMemoryEffectFree(&op) && op.getNumRegions() == 0 &&
        op.getNumSuccessors() == 0 &&
        llvm::all_of(op.getAttrs(), [](NamedAttribute attr) {
          return attr.getNameDialect() != nullptr ||
                 attr.getName() == "twoState";
        })) {
      // Lower this operation to a balanced binary tree of the same operation.
      SmallVector<Operation *> newOps;
      auto result = lowerFullyAssociativeOp(op, op.getOperands(), newOps);
      op.getResult(0).replaceAllUsesWith(result);
      op.erase();

      // Make sure we revisit the newly inserted operations.
      opIterator = Block::iterator(newOps.front());
      continue;
    }

    // Turn a + -cst  ==> a - cst
    if (auto addOp = dyn_cast<comb::AddOp>(op)) {
      if (auto cst = addOp.getOperand(1).getDefiningOp<hw::ConstantOp>()) {
        assert(addOp.getNumOperands() == 2 && "commutative lowering is done");
        if (cst.getValue().isNegative()) {
          Operation *firstOp = rewriteAddWithNegativeConstant(addOp, cst);
          opIterator = Block::iterator(firstOp);
          continue;
        }
      }
    }

    // Lower hw.struct_explode ops into a set of hw.struct_extract ops which
    // have well-defined SV emission semantics.
    if (auto structExplodeOp = dyn_cast<hw::StructExplodeOp>(op)) {
      Operation *firstOp = lowerStructExplodeOp(structExplodeOp);
      opIterator = Block::iterator(firstOp);
      continue;
    }

    // Try to anticipate expressions that ExportVerilog may spill to a temporary
    // inout, and re-use an existing inout when possible. This is legal when op
    // is an expression in a non-procedural region.
    if (!isProceduralRegion && isVerilogExpression(&op))
      (void)reuseExistingInOut(&op, options);
  }

  if (isProceduralRegion) {
    // If there is no operation, there is nothing to do.
    if (block.empty())
      return success();

    // In a procedural region, logic operations needs to be top of blocks so
    // mvoe logic operations to valid program points.

    // This keeps tracks of an insertion point of logic op.
    std::pair<Block *, Block::iterator> logicOpInsertionPoint =
        findLogicOpInsertionPoint(&block.front());
    for (auto &op : llvm::make_early_inc_range(block)) {
      if (auto logic = dyn_cast<LogicOp>(&op)) {
        // If the logic op is already located at the given point, increment the
        // iterator to keep the order of logic operations in the block.
        if (logicOpInsertionPoint.second == logic->getIterator()) {
          ++logicOpInsertionPoint.second;
          continue;
        }
        // Otherwise, move the op to the insertion point.
        logic->moveBefore(logicOpInsertionPoint.first,
                          logicOpInsertionPoint.second);
      }
    }
    return success();
  }

  // Now that all the basic ops are settled, check for any use-before def issues
  // in graph regions.  Lower these into explicit wires to keep the emitter
  // simple.

  SmallPtrSet<Operation *, 32> seenOperations;

  for (auto &op : llvm::make_early_inc_range(block)) {
    // Do not reorder LTL expressions, which are always emitted inline.
    if (isa<ltl::LTLDialect>(op.getDialect()))
      continue;

    // Check the users of any expressions to see if they are
    // lexically below the operation itself.  If so, it is being used out
    // of order.
    bool haveAnyOutOfOrderUses = false;
    for (auto *userOp : op.getUsers()) {
      // If the user is in a suboperation like an always block, then zip up
      // to the operation that uses it.
      while (&block != &userOp->getParentRegion()->front())
        userOp = userOp->getParentOp();

      if (seenOperations.count(userOp)) {
        haveAnyOutOfOrderUses = true;
        break;
      }
    }

    // Remember that we've seen this operation.
    seenOperations.insert(&op);

    // If all the uses of the operation are below this, then we're ok.
    if (!haveAnyOutOfOrderUses)
      continue;

    // If this is a reg/wire declaration, then we move it to the top of the
    // block.  We can't abstract the inout result.
    if (isMovableDeclaration(&op)) {
      op.moveBefore(&block.front());
      continue;
    }

    // If this is a constant, then we move it to the top of the block.
    if (isConstantExpression(&op)) {
      op.moveBefore(&block.front());
      continue;
    }

    // If this is a an operation reading from a declaration, move it up,
    // along with the corresponding declaration.
    if (auto readInOut = dyn_cast<ReadInOutOp>(op)) {
      auto *def = readInOut.getInput().getDefiningOp();
      if (isMovableDeclaration(def)) {
        op.moveBefore(&block.front());
        def->moveBefore(&block.front());
        continue;
      }
    }

    // Otherwise, we need to lower this to a wire to resolve this.
    lowerUsersToTemporaryWire(op,
                              /*emitWireAtBlockBegin=*/true);
  }
  return success();
}

// NOLINTNEXTLINE(misc-no-recursion)
LogicalResult ExportVerilog::prepareHWModule(hw::HWModuleOp module,
                                             const LoweringOptions &options) {
  // Zero-valued logic pruning.
  pruneZeroValuedLogic(module);

  // Legalization.
  if (failed(legalizeHWModule(*module.getBodyBlock(), options)))
    return failure();

  EmittedExpressionStateManager expressionStateManager(options);
  // Spill wires to prettify verilog outputs.
  prettifyAfterLegalization(*module.getBodyBlock(), expressionStateManager);
  return success();
}

namespace {

struct PrepareForEmissionPass
    : public PrepareForEmissionBase<PrepareForEmissionPass> {
  void runOnOperation() override {
    HWModuleOp module = getOperation();
    LoweringOptions options(cast<mlir::ModuleOp>(module->getParentOp()));
    if (failed(prepareHWModule(module, options)))
      signalPassFailure();
  }
};

} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::createPrepareForEmissionPass() {
  return std::make_unique<PrepareForEmissionPass>();
}

//===- CalyxLoweringUtils.h - Calyx lowering utility methods ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines various lowering utility methods for converting to
// and from Calyx programs.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CALYX_CALYXLOWERINGUTILS_H
#define CIRCT_DIALECT_CALYX_CALYXLOWERINGUTILS_H

#include "circt/Dialect/Calyx/CalyxHelpers.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include <variant>

namespace circt {
namespace calyx {

void appendPortsForExternalMemref(PatternRewriter &rewriter, StringRef memName,
                                  Value memref, unsigned memoryID,
                                  SmallVectorImpl<calyx::PortInfo> &inPorts,
                                  SmallVectorImpl<calyx::PortInfo> &outPorts);

// Walks the control of this component, and appends source information for leaf
// nodes. It also appends a position attribute that connects the source location
// metadata to the corresponding control operation.
WalkResult
getCiderSourceLocationMetadata(calyx::ComponentOp component,
                               SmallVectorImpl<Attribute> &sourceLocations);

// Tries to match a constant value defined by op. If the match was
// successful, returns true and binds the constant to 'value'. If unsuccessful,
// the value is unmodified.
bool matchConstantOp(Operation *op, APInt &value);

// Returns true if there exists only a single memref::LoadOp which loads from
// the memory referenced by loadOp.
bool singleLoadFromMemory(Value memoryReference);

// Returns true if there are no memref::StoreOp uses with the referenced
// memory.
bool noStoresToMemory(Value memoryReference);

// Get the index'th output port of compOp.
Value getComponentOutput(calyx::ComponentOp compOp, unsigned outPortIdx);

// If the provided type is an index type, converts it to i32, else, returns the
// unmodified type.
Type convIndexType(OpBuilder &builder, Type type);

// Creates a new calyx::CombGroupOp or calyx::GroupOp group within compOp.
template <typename TGroup>
TGroup createGroup(OpBuilder &builder, calyx::ComponentOp compOp, Location loc,
                   Twine uniqueName) {
  mlir::IRRewriter::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(compOp.getWiresOp().getBodyBlock());
  return builder.create<TGroup>(loc, uniqueName.str());
}

/// Creates register assignment operations within the provided groupOp.
/// The component operation will house the constants.
void buildAssignmentsForRegisterWrite(OpBuilder &builder,
                                      calyx::GroupOp groupOp,
                                      calyx::ComponentOp componentOp,
                                      calyx::RegisterOp &reg, Value inputValue);

// A structure representing a set of ports which act as a memory interface for
// external memories.
struct MemoryPortsImpl {
  std::optional<Value> readData;
  std::optional<Value> readEn;
  std::optional<Value> readDone;
  std::optional<Value> writeData;
  std::optional<Value> writeEn;
  std::optional<Value> writeDone;
  SmallVector<Value> addrPorts;
};

// Represents the interface of memory in Calyx. The various lowering passes
// are agnostic wrt. whether working with a calyx::MemoryOp (internally
// allocated memory) or MemoryPortsImpl (external memory).
struct MemoryInterface {
  MemoryInterface();
  explicit MemoryInterface(const MemoryPortsImpl &ports);
  explicit MemoryInterface(calyx::MemoryOp memOp);
  explicit MemoryInterface(calyx::SeqMemoryOp memOp);

  // Getter methods for each memory interface port.
  Value readData();
  Value readEn();
  Value readDone();
  Value writeData();
  Value writeEn();
  Value writeDone();
  std::optional<Value> readDataOpt();
  std::optional<Value> readEnOpt();
  std::optional<Value> readDoneOpt();
  std::optional<Value> writeDataOpt();
  std::optional<Value> writeEnOpt();
  std::optional<Value> writeDoneOpt();
  ValueRange addrPorts();

private:
  std::variant<calyx::MemoryOp, calyx::SeqMemoryOp, MemoryPortsImpl> impl;
};

// A common interface for any loop operation that needs to be lowered to Calyx.
class BasicLoopInterface {
public:
  virtual ~BasicLoopInterface();

  // Returns the arguments to this loop operation.
  virtual Block::BlockArgListType getBodyArgs() = 0;

  // Returns body of this loop operation.
  virtual Block *getBodyBlock() = 0;

  // Returns the location of the loop interface.
  virtual Location getLoc() = 0;

  // Returns the number of iterations the loop will conduct if known.
  virtual std::optional<int64_t> getBound() = 0;
};

// A common interface for loop operations that have conditionals (e.g., while
// loops) that need to be lowered to Calyx.
class LoopInterface : BasicLoopInterface {
public:
  // Returns the Block in which the condition exists.
  virtual Block *getConditionBlock() = 0;

  // Returns the condition as a Value.
  virtual Value getConditionValue() = 0;
};

// Provides an interface for the control flow `while` operation across different
// dialects.
template <typename T>
class WhileOpInterface : LoopInterface {
  static_assert(std::is_convertible_v<T, Operation *>);

public:
  explicit WhileOpInterface(T op) : impl(op) {}
  explicit WhileOpInterface(Operation *op) : impl(dyn_cast_or_null<T>(op)) {}

  // Returns the operation.
  T getOperation() { return impl; }

  // Returns the source location of the operation.
  Location getLoc() override { return impl->getLoc(); }

private:
  T impl;
};

// Provides an interface for the control flow `forOp` operation across different
// dialects.
template <typename T>
class RepeatOpInterface : BasicLoopInterface {
  static_assert(std::is_convertible_v<T, Operation *>);

public:
  explicit RepeatOpInterface(T op) : impl(op) {}
  explicit RepeatOpInterface(Operation *op) : impl(dyn_cast_or_null<T>(op)) {}

  // Returns the operation.
  T getOperation() { return impl; }

  // Returns the source location of the operation.
  Location getLoc() override { return impl->getLoc(); }

private:
  T impl;
};

/// Holds common utilities used for scheduling when lowering to Calyx.
template <typename T>
class SchedulerInterface {
public:
  /// Register 'scheduleable' as being generated through lowering 'block'.
  ///
  /// TODO(mortbopet): Add a post-insertion check to ensure that the use-def
  /// ordering invariant holds for the groups. When the control schedule is
  /// generated, scheduleables within a block are emitted sequentially based on
  /// the order that this function was called during conversion.
  ///
  /// Currently, we assume this to always be true. Walking the FuncOp IR implies
  /// sequential iteration over operations within basic blocks.
  void addBlockScheduleable(mlir::Block *block, const T &scheduleable) {
    blockScheduleables[block].push_back(scheduleable);
  }

  /// Returns an ordered list of schedulables which registered themselves to be
  /// a result of lowering the block in the source program. The list order
  /// follows def-use chains between the scheduleables in the block.
  SmallVector<T> getBlockScheduleables(mlir::Block *block) {
    if (auto it = blockScheduleables.find(block);
        it != blockScheduleables.end())
      return it->second;
    /// In cases of a block resulting in purely combinational logic, no
    /// scheduleables registered themselves with the block.
    return {};
  }

private:
  /// BlockScheduleables is a list of scheduleables that should be
  /// sequentially executed when executing the associated basic block.
  DenseMap<mlir::Block *, SmallVector<T>> blockScheduleables;
};

//===----------------------------------------------------------------------===//
// Lowering state classes
//===----------------------------------------------------------------------===//

// Handles state during the lowering of a loop. It will be used for
// several lowering patterns.
template <typename Loop>
class LoopLoweringStateInterface {
  static_assert(std::is_base_of_v<BasicLoopInterface, Loop>);

public:
  ~LoopLoweringStateInterface() = default;

  /// Register reg as being the idx'th iter_args register for 'op'.
  void addLoopIterReg(Loop op, calyx::RegisterOp reg, unsigned idx) {
    assert(loopIterRegs[op.getOperation()].count(idx) == 0 &&
           "A register was already registered for the given loop iter_arg "
           "index");
    assert(idx < op.getBodyArgs().size());
    loopIterRegs[op.getOperation()][idx] = reg;
  }

  /// Return a mapping of block argument indices to block argument.
  calyx::RegisterOp getLoopIterReg(Loop op, unsigned idx) {
    auto iterRegs = getLoopIterRegs(op);
    auto it = iterRegs.find(idx);
    assert(it != iterRegs.end() &&
           "No iter arg register set for the provided index");
    return it->second;
  }

  /// Return a mapping of block argument indices to block argument.
  const DenseMap<unsigned, calyx::RegisterOp> &getLoopIterRegs(Loop op) {
    return loopIterRegs[op.getOperation()];
  }

  /// Registers grp to be the loop latch group of `op`.
  void setLoopLatchGroup(Loop op, calyx::GroupOp group) {
    Operation *operation = op.getOperation();
    assert(loopLatchGroups.count(operation) == 0 &&
           "A latch group was already set for this loopOp");
    loopLatchGroups[operation] = group;
  }

  /// Retrieve the loop latch group registered for `op`.
  calyx::GroupOp getLoopLatchGroup(Loop op) {
    auto it = loopLatchGroups.find(op.getOperation());
    assert(it != loopLatchGroups.end() &&
           "No loop latch group was set for this loopOp");
    return it->second;
  }

  /// Registers groups to be the loop init groups of `op`.
  void setLoopInitGroups(Loop op, SmallVector<calyx::GroupOp> groups) {
    Operation *operation = op.getOperation();
    assert(loopInitGroups.count(operation) == 0 &&
           "Init group(s) was already set for this loopOp");
    loopInitGroups[operation] = std::move(groups);
  }

  /// Retrieve the loop init groups registered for `op`.
  SmallVector<calyx::GroupOp> getLoopInitGroups(Loop op) {
    auto it = loopInitGroups.find(op.getOperation());
    assert(it != loopInitGroups.end() &&
           "No init group(s) was set for this loopOp");
    return it->second;
  }

  /// Creates a new group that assigns the 'ops' values to the iter arg
  /// registers of the loop operation.
  calyx::GroupOp buildLoopIterArgAssignments(OpBuilder &builder, Loop op,
                                             calyx::ComponentOp componentOp,
                                             Twine uniqueSuffix,
                                             MutableArrayRef<OpOperand> ops) {
    /// Pass iteration arguments through registers. This follows closely
    /// to what is done for branch ops.
    std::string groupName = "assign_" + uniqueSuffix.str();
    auto groupOp = calyx::createGroup<calyx::GroupOp>(builder, componentOp,
                                                      op.getLoc(), groupName);
    /// Create register assignment for each iter_arg. a calyx::GroupDone signal
    /// is created for each register. These will be &'ed together in
    /// MultipleGroupDonePattern.
    for (OpOperand &arg : ops) {
      auto reg = getLoopIterReg(op, arg.getOperandNumber());
      buildAssignmentsForRegisterWrite(builder, groupOp, componentOp, reg,
                                       arg.get());
    }
    return groupOp;
  }

private:
  /// A mapping from loop ops to iteration argument registers.
  DenseMap<Operation *, DenseMap<unsigned, calyx::RegisterOp>> loopIterRegs;

  /// A loop latch group is a group that should be sequentially executed when
  /// finishing a loop body. The execution of this group will write the
  /// yield'ed loop body values to the iteration argument registers.
  DenseMap<Operation *, calyx::GroupOp> loopLatchGroups;

  /// Loop init groups are to be scheduled before the while operation. These
  /// groups should set the initial value(s) of the loop init_args register(s).
  DenseMap<Operation *, SmallVector<calyx::GroupOp>> loopInitGroups;
};

// Handles state during the lowering of a Calyx component. This provides common
// tools for converting to the Calyx ComponentOp.
class ComponentLoweringStateInterface {
public:
  ComponentLoweringStateInterface(calyx::ComponentOp component);

  ~ComponentLoweringStateInterface();

  /// Returns the calyx::ComponentOp associated with this lowering state.
  calyx::ComponentOp getComponentOp();

  /// Register reg as being the idx'th argument register for block. This is
  /// necessary for the `BuildBBReg` pass.
  void addBlockArgReg(Block *block, calyx::RegisterOp reg, unsigned idx);

  /// Return a mapping of block argument indices to block argument registers.
  /// This is necessary for the `BuildBBReg` pass.
  const DenseMap<unsigned, calyx::RegisterOp> &getBlockArgRegs(Block *block);

  /// Register 'grp' as a group which performs block argument
  /// register transfer when transitioning from basic block 'from' to 'to'.
  void addBlockArgGroup(Block *from, Block *to, calyx::GroupOp grp);

  /// Returns a list of groups to be evaluated to perform the block argument
  /// register assignments when transitioning from basic block 'from' to 'to'.
  ArrayRef<calyx::GroupOp> getBlockArgGroups(Block *from, Block *to);

  /// Returns a unique name within compOp with the provided prefix.
  std::string getUniqueName(StringRef prefix);

  /// Returns a unique name associated with a specific operation.
  StringRef getUniqueName(Operation *op);

  /// Registers a unique name for a given operation using a provided prefix.
  void setUniqueName(Operation *op, StringRef prefix);

  /// Register value v as being evaluated when scheduling group.
  void registerEvaluatingGroup(Value v, calyx::GroupInterface group);

  /// Register reg as being the idx'th return value register.
  void addReturnReg(calyx::RegisterOp reg, unsigned idx);

  /// Returns the idx'th return value register.
  calyx::RegisterOp getReturnReg(unsigned idx);

  /// Registers a memory interface as being associated with a memory identified
  /// by 'memref'.
  void registerMemoryInterface(Value memref,
                               const calyx::MemoryInterface &memoryInterface);

  /// Returns the memory interface registered for the given memref.
  calyx::MemoryInterface getMemoryInterface(Value memref);

  /// If v is an input to any memory registered within this component, returns
  /// the memory. If not, returns null.
  std::optional<calyx::MemoryInterface> isInputPortOfMemory(Value v);

  /// Assign a mapping between the source funcOp result indices and the
  /// corresponding output port indices of this componentOp.
  void setFuncOpResultMapping(const DenseMap<unsigned, unsigned> &mapping);

  /// Get the output port index of this component for which the funcReturnIdx of
  /// the original function maps to.
  unsigned getFuncOpResultMapping(unsigned funcReturnIdx);

  /// Return the group which evaluates the value v. Optionally, caller may
  /// specify the expected type of the group.
  template <typename TGroupOp = calyx::GroupInterface>
  TGroupOp getEvaluatingGroup(Value v) {
    auto it = valueGroupAssigns.find(v);
    assert(it != valueGroupAssigns.end() && "No group evaluating value!");
    if constexpr (std::is_same_v<TGroupOp, calyx::GroupInterface>)
      return it->second;
    else {
      auto group = dyn_cast<TGroupOp>(it->second.getOperation());
      assert(group && "Actual group type differed from expected group type");
      return group;
    }
  }

  template <typename TLibraryOp>
  TLibraryOp getNewLibraryOpInstance(OpBuilder &builder, Location loc,
                                     TypeRange resTypes) {
    mlir::IRRewriter::InsertionGuard guard(builder);
    Block *body = component.getBodyBlock();
    builder.setInsertionPoint(body, body->begin());
    auto name = TLibraryOp::getOperationName().split(".").second;
    return builder.create<TLibraryOp>(loc, getUniqueName(name), resTypes);
  }

private:
  /// The component which this lowering state is associated to.
  calyx::ComponentOp component;

  /// A mapping from blocks to block argument registers.
  DenseMap<Block *, DenseMap<unsigned, calyx::RegisterOp>> blockArgRegs;

  /// Block arg groups is a list of groups that should be sequentially
  /// executed when passing control from the source to destination block.
  /// Block arg groups are executed before blockScheduleables (akin to a
  /// phi-node).
  DenseMap<Block *, DenseMap<Block *, SmallVector<calyx::GroupOp>>>
      blockArgGroups;

  /// A mapping of string prefixes and the current uniqueness counter for that
  /// prefix. Used to generate unique names.
  std::map<std::string, unsigned> prefixIdMap;

  /// A mapping from Operations and previously assigned unique name of the op.
  std::map<Operation *, std::string> opNames;

  /// A mapping between SSA values and the groups which assign them.
  DenseMap<Value, calyx::GroupInterface> valueGroupAssigns;

  /// A mapping from return value indexes to return value registers.
  DenseMap<unsigned, calyx::RegisterOp> returnRegs;

  /// A mapping from memref's to their corresponding Calyx memory interface.
  DenseMap<Value, calyx::MemoryInterface> memories;

  /// A mapping between the source funcOp result indices and the corresponding
  /// output port indices of this componentOp.
  DenseMap<unsigned, unsigned> funcOpResultMapping;
};

/// An interface for conversion passes that lower Calyx programs. This handles
/// state during the lowering of a Calyx program.
class CalyxLoweringState {
public:
  explicit CalyxLoweringState(mlir::ModuleOp module,
                              StringRef topLevelFunction);

  /// Returns the current program.
  mlir::ModuleOp getModule();

  /// Returns the name of the top-level function in the source program.
  StringRef getTopLevelFunction() const;

  /// Returns a meaningful name for a block within the program scope (removes
  /// the ^ prefix from block names).
  std::string blockName(Block *b);

  /// Returns the component lowering state associated with `op`. If not found
  /// already found, a new mapping is added for this ComponentOp. Different
  /// conversions may have different derived classes of the interface, so we
  /// provided a template.
  template <typename T = calyx::ComponentLoweringStateInterface>
  T *getState(calyx::ComponentOp op) {
    static_assert(std::is_convertible_v<T, ComponentLoweringStateInterface>);
    auto it = componentStates.find(op);
    if (it == componentStates.end()) {
      // Create a new ComponentLoweringState for the compOp.
      bool success;
      std::tie(it, success) =
          componentStates.try_emplace(op, std::make_unique<T>(op));
    }

    return static_cast<T *>(it->second.get());
  }

  /// Returns a meaningful name for a value within the program scope.
  template <typename ValueOrBlock>
  std::string irName(ValueOrBlock &v) {
    std::string s;
    llvm::raw_string_ostream os(s);
    mlir::AsmState asmState(module);
    v.printAsOperand(os, asmState);
    return s;
  }

private:
  /// The name of this top-level function.
  StringRef topLevelFunction;
  /// The program associated with this state.
  mlir::ModuleOp module;
  /// Mapping from ComponentOp to component lowering state.
  DenseMap<Operation *, std::unique_ptr<ComponentLoweringStateInterface>>
      componentStates;
};

/// Extra state that is passed to all PartialLoweringPatterns so they can record
/// when they have run on an Operation, and only run once.
using PatternApplicationState =
    DenseMap<const mlir::RewritePattern *, SmallPtrSet<Operation *, 16>>;

/// Base class for partial lowering passes. A partial lowering pass
/// modifies the root operation in place, but does not replace the root
/// operation.
/// The RewritePatternType template parameter allows for using both
/// OpRewritePattern (default) or OpInterfaceRewritePattern.
template <class OpType,
          template <class> class RewritePatternType = OpRewritePattern>
class PartialLoweringPattern : public RewritePatternType<OpType> {
public:
  using RewritePatternType<OpType>::RewritePatternType;
  PartialLoweringPattern(MLIRContext *ctx, LogicalResult &resRef,
                         PatternApplicationState &patternState)
      : RewritePatternType<OpType>(ctx), partialPatternRes(resRef),
        patternState(patternState) {}

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    // If this pattern has been applied to this op, it should now fail to match.
    if (patternState[this].contains(op))
      return failure();

    // Do the actual rewrite, marking this op as updated. Because the op is
    // marked as updated, the pattern driver will re-enqueue the op again.
    rewriter.updateRootInPlace(
        op, [&] { partialPatternRes = partiallyLower(op, rewriter); });

    // Mark that this pattern has been applied to this op.
    patternState[this].insert(op);

    return partialPatternRes;
  }

  // Hook for subclasses to lower the op using the rewriter.
  //
  // Note that this call is wrapped in `updateRootInPlace`, so any direct IR
  // mutations that are legal to apply during a root update of op are allowed.
  //
  // Also note that this means the op will be re-enqueued to the greedy
  // rewriter's worklist. A safeguard is in place to prevent patterns from
  // running multiple times, but if the op is erased or otherwise becomes dead
  // after the call to `partiallyLower`, there will likely be use-after-free
  // violations. If you will erase the op, override `matchAndRewrite` directly.
  virtual LogicalResult partiallyLower(OpType op,
                                       PatternRewriter &rewriter) const = 0;

private:
  LogicalResult &partialPatternRes;
  PatternApplicationState &patternState;
};

/// Helper to update the top-level ModuleOp to set the entrypoing function.
LogicalResult applyModuleOpConversion(mlir::ModuleOp,
                                      StringRef topLevelFunction);

/// FuncOpPartialLoweringPatterns are patterns which intend to match on FuncOps
/// and then perform their own walking of the IR.
class FuncOpPartialLoweringPattern
    : public calyx::PartialLoweringPattern<mlir::func::FuncOp> {

public:
  FuncOpPartialLoweringPattern(
      MLIRContext *context, LogicalResult &resRef,
      PatternApplicationState &patternState,
      DenseMap<mlir::func::FuncOp, calyx::ComponentOp> &map,
      calyx::CalyxLoweringState &state);

  /// Entry point to initialize the state of this class and conduct the partial
  /// lowering.
  LogicalResult partiallyLower(mlir::func::FuncOp funcOp,
                               PatternRewriter &rewriter) const override final;

  /// Returns the component operation associated with the currently executing
  /// partial lowering.
  calyx::ComponentOp getComponent() const;

  // Returns the component state associated with the currently executing
  // partial lowering.
  template <typename T = ComponentLoweringStateInterface>
  T &getState() const {
    static_assert(
        std::is_convertible_v<T, calyx::ComponentLoweringStateInterface>);
    assert(
        componentLoweringState != nullptr &&
        "Component lowering state should be set during pattern construction");
    return *static_cast<T *>(componentLoweringState);
  }

  /// Return the calyx lowering state for this pattern.
  CalyxLoweringState &loweringState() const;

  // Hook for subclasses to lower the op using the rewriter.
  //
  // Note that this call is wrapped in `updateRootInPlace`, so any direct IR
  // mutations that are legal to apply during a root update of op are allowed.
  //
  // Also note that this means the op will be re-enqueued to the greedy
  // rewriter's worklist. A safeguard is in place to prevent patterns from
  // running multiple times, but if the op is erased or otherwise becomes dead
  // after the call to `partiallyLower`, there will likely be use-after-free
  // violations. If you will erase the op, override `matchAndRewrite` directly.
  virtual LogicalResult
  partiallyLowerFuncToComp(mlir::func::FuncOp funcOp,
                           PatternRewriter &rewriter) const = 0;

protected:
  // A map from FuncOp to it's respective ComponentOp lowering.
  DenseMap<mlir::func::FuncOp, calyx::ComponentOp> &functionMapping;

private:
  mutable ComponentOp componentOp;
  mutable ComponentLoweringStateInterface *componentLoweringState = nullptr;
  CalyxLoweringState &calyxLoweringState;
};

/// Converts all index-typed operations and values to i32 values.
class ConvertIndexTypes : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(mlir::func::FuncOp funcOp,
                           PatternRewriter &rewriter) const override;
};

/// GroupDoneOp's are terminator operations and should therefore be the last
/// operator in a group. During group construction, we always append assignments
/// to the end of a group, resulting in group_done ops migrating away from the
/// terminator position. This pattern moves such ops to the end of their group.
struct NonTerminatingGroupDonePattern
    : mlir::OpRewritePattern<calyx::GroupDoneOp> {
  using mlir::OpRewritePattern<calyx::GroupDoneOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(calyx::GroupDoneOp groupDoneOp,
                                PatternRewriter &) const override;
};

/// When building groups which contain accesses to multiple sequential
/// components, a group_done op is created for each of these. This pattern
/// and's each of the group_done values into a single group_done.
struct MultipleGroupDonePattern : mlir::OpRewritePattern<calyx::GroupOp> {
  using mlir::OpRewritePattern<calyx::GroupOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(calyx::GroupOp groupOp,
                                PatternRewriter &rewriter) const override;
};

/// Removes calyx::CombGroupOps which are unused. These correspond to
/// combinational groups created during op building that, after conversion,
/// have either been inlined into calyx::GroupOps or are referenced by an
/// if/while with statement.
/// We do not eliminate unused calyx::GroupOps; this should never happen, and is
/// considered an error. In these cases, the program will be invalidated when
/// the Calyx verifiers execute.
struct EliminateUnusedCombGroups : mlir::OpRewritePattern<calyx::CombGroupOp> {
  using mlir::OpRewritePattern<calyx::CombGroupOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(calyx::CombGroupOp combGroupOp,
                                PatternRewriter &rewriter) const override;
};

/// This pass recursively inlines use-def chains of combinational logic (from
/// non-stateful groups) into groups referenced in the control schedule.
class InlineCombGroups
    : public calyx::PartialLoweringPattern<calyx::GroupInterface,
                                           mlir::OpInterfaceRewritePattern> {
public:
  InlineCombGroups(MLIRContext *context, LogicalResult &resRef,
                   PatternApplicationState &patternState,
                   calyx::CalyxLoweringState &pls);

  LogicalResult partiallyLower(calyx::GroupInterface originGroup,
                               PatternRewriter &rewriter) const override;

private:
  void
  recurseInlineCombGroups(PatternRewriter &rewriter,
                          ComponentLoweringStateInterface &state,
                          llvm::SmallSetVector<Operation *, 8> &inlinedGroups,
                          calyx::GroupInterface originGroup,
                          calyx::GroupInterface recGroup, bool doInline) const;

  calyx::CalyxLoweringState &cls;
};

/// This pass rewrites memory accesses that have a width mismatch. Such
/// mismatches are due to index types being assumed 32-bit wide due to the lack
/// of a width inference pass.
class RewriteMemoryAccesses
    : public calyx::PartialLoweringPattern<calyx::AssignOp> {
public:
  RewriteMemoryAccesses(MLIRContext *context, LogicalResult &resRef,
                        PatternApplicationState &patternState,
                        calyx::CalyxLoweringState &cls)
      : PartialLoweringPattern(context, resRef, patternState), cls(cls) {}

  LogicalResult partiallyLower(calyx::AssignOp assignOp,
                               PatternRewriter &rewriter) const override;

private:
  calyx::CalyxLoweringState &cls;
};

/// Builds registers for each block argument in the program.
class BuildBasicBlockRegs : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(mlir::func::FuncOp funcOp,
                           PatternRewriter &rewriter) const override;
};

/// Builds registers for the return statement of the program and constant
/// assignments to the component return value.
class BuildReturnRegs : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(mlir::func::FuncOp funcOp,
                           PatternRewriter &rewriter) const override;
};

} // namespace calyx
} // namespace circt

#endif // CIRCT_DIALECT_CALYX_CALYXLOWERINGUTILS_H

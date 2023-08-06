//===- CalyxLoweringUtils.cpp - Calyx lowering utility methods --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Various lowering utility methods converting to and from Calyx programs.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Calyx/CalyxLoweringUtils.h"
#include "circt/Dialect/Calyx/CalyxHelpers.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"

#include <variant>

using namespace llvm;
using namespace mlir;
using namespace mlir::arith;

namespace circt {
namespace calyx {

void appendPortsForExternalMemref(PatternRewriter &rewriter, StringRef memName,
                                  Value memref, unsigned memoryID,
                                  SmallVectorImpl<calyx::PortInfo> &inPorts,
                                  SmallVectorImpl<calyx::PortInfo> &outPorts) {
  MemRefType memrefType = memref.getType().cast<MemRefType>();

  // Ports constituting a memory interface are added a set of attributes under
  // a "mem : {...}" dictionary. These attributes allows for deducing which
  // top-level I/O signals constitutes a unique memory interface.
  auto getMemoryInterfaceAttr = [&](StringRef tag,
                                    std::optional<unsigned> addrIdx = {}) {
    auto attrs = SmallVector<NamedAttribute>{
        // "id" denotes a unique memory interface.
        rewriter.getNamedAttr("id", rewriter.getI32IntegerAttr(memoryID)),
        // "tag" denotes the function of this signal.
        rewriter.getNamedAttr("tag", rewriter.getStringAttr(tag))};
    if (addrIdx.has_value())
      // "addr_idx" denotes the address index of this signal, for
      // multi-dimensional memory interfaces.
      attrs.push_back(rewriter.getNamedAttr(
          "addr_idx", rewriter.getI32IntegerAttr(*addrIdx)));

    return rewriter.getNamedAttr("mem", rewriter.getDictionaryAttr(attrs));
  };

  // Read data
  inPorts.push_back(calyx::PortInfo{
      rewriter.getStringAttr(memName + "_read_data"),
      memrefType.getElementType(), calyx::Direction::Input,
      DictionaryAttr::get(rewriter.getContext(),
                          {getMemoryInterfaceAttr("read_data")})});

  // Done
  inPorts.push_back(
      calyx::PortInfo{rewriter.getStringAttr(memName + "_done"),
                      rewriter.getI1Type(), calyx::Direction::Input,
                      DictionaryAttr::get(rewriter.getContext(),
                                          {getMemoryInterfaceAttr("done")})});

  // Write data
  outPorts.push_back(calyx::PortInfo{
      rewriter.getStringAttr(memName + "_write_data"),
      memrefType.getElementType(), calyx::Direction::Output,
      DictionaryAttr::get(rewriter.getContext(),
                          {getMemoryInterfaceAttr("write_data")})});

  // Memory address outputs
  for (auto dim : enumerate(memrefType.getShape())) {
    outPorts.push_back(calyx::PortInfo{
        rewriter.getStringAttr(memName + "_addr" + std::to_string(dim.index())),
        rewriter.getIntegerType(calyx::handleZeroWidth(dim.value())),
        calyx::Direction::Output,
        DictionaryAttr::get(rewriter.getContext(),
                            {getMemoryInterfaceAttr("addr", dim.index())})});
  }

  // Write enable
  outPorts.push_back(calyx::PortInfo{
      rewriter.getStringAttr(memName + "_write_en"), rewriter.getI1Type(),
      calyx::Direction::Output,
      DictionaryAttr::get(rewriter.getContext(),
                          {getMemoryInterfaceAttr("write_en")})});
}

WalkResult
getCiderSourceLocationMetadata(calyx::ComponentOp component,
                               SmallVectorImpl<Attribute> &sourceLocations) {
  Builder builder(component->getContext());
  return component.getControlOp().walk([&](Operation *op) {
    if (!calyx::isControlLeafNode(op))
      return WalkResult::advance();

    std::string sourceLocation;
    llvm::raw_string_ostream os(sourceLocation);
    op->getLoc()->print(os);
    int64_t position = sourceLocations.size();
    sourceLocations.push_back(
        StringAttr::get(op->getContext(), sourceLocation));

    op->setAttr("pos", builder.getI64IntegerAttr(position));
    return WalkResult::advance();
  });
}

bool matchConstantOp(Operation *op, APInt &value) {
  return mlir::detail::constant_int_op_binder(&value).match(op);
}

bool singleLoadFromMemory(Value memoryReference) {
  return llvm::count_if(memoryReference.getUses(), [](OpOperand &user) {
           return isa<mlir::memref::LoadOp>(user.getOwner());
         }) <= 1;
}

bool noStoresToMemory(Value memoryReference) {
  return llvm::none_of(memoryReference.getUses(), [](OpOperand &user) {
    return isa<mlir::memref::StoreOp>(user.getOwner());
  });
}

Value getComponentOutput(calyx::ComponentOp compOp, unsigned outPortIdx) {
  size_t index = compOp.getInputPortInfo().size() + outPortIdx;
  assert(index < compOp.getNumArguments() &&
         "Exceeded number of arguments in the Component");
  return compOp.getArgument(index);
}

Type convIndexType(OpBuilder &builder, Type type) {
  if (type.isIndex())
    return builder.getI32Type();
  return type;
}

void buildAssignmentsForRegisterWrite(OpBuilder &builder,
                                      calyx::GroupOp groupOp,
                                      calyx::ComponentOp componentOp,
                                      calyx::RegisterOp &reg,
                                      Value inputValue) {
  mlir::IRRewriter::InsertionGuard guard(builder);
  auto loc = inputValue.getLoc();
  builder.setInsertionPointToEnd(groupOp.getBodyBlock());
  builder.create<calyx::AssignOp>(loc, reg.getIn(), inputValue);
  builder.create<calyx::AssignOp>(
      loc, reg.getWriteEn(), createConstant(loc, builder, componentOp, 1, 1));
  builder.create<calyx::GroupDoneOp>(loc, reg.getDone());
}

//===----------------------------------------------------------------------===//
// MemoryInterface
//===----------------------------------------------------------------------===//

MemoryInterface::MemoryInterface() = default;
MemoryInterface::MemoryInterface(const MemoryPortsImpl &ports) : impl(ports) {}
MemoryInterface::MemoryInterface(calyx::MemoryOp memOp) : impl(memOp) {}
MemoryInterface::MemoryInterface(calyx::SeqMemoryOp memOp) : impl(memOp) {}

Value MemoryInterface::readData() {
  auto readData = readDataOpt();
  assert(readData.has_value() && "Memory does not have readData");
  return readData.value();
}

Value MemoryInterface::readEn() {
  auto readEn = readEnOpt();
  assert(readEn.has_value() && "Memory does not have readEn");
  return readEn.value();
}

Value MemoryInterface::readDone() {
  auto readDone = readDoneOpt();
  assert(readDone.has_value() && "Memory does not have readDone");
  return readDone.value();
}

Value MemoryInterface::writeData() {
  auto writeData = writeDataOpt();
  assert(writeData.has_value() && "Memory does not have writeData");
  return writeData.value();
}

Value MemoryInterface::writeEn() {
  auto writeEn = writeEnOpt();
  assert(writeEn.has_value() && "Memory does not have writeEn");
  return writeEn.value();
}

Value MemoryInterface::writeDone() {
  auto writeDone = writeDoneOpt();
  assert(writeDone.has_value() && "Memory doe snot have writeDone");
  return writeDone.value();
}

std::optional<Value> MemoryInterface::readDataOpt() {
  if (auto *memOp = std::get_if<calyx::MemoryOp>(&impl); memOp) {
    return memOp->readData();
  }

  if (auto *memOp = std::get_if<calyx::SeqMemoryOp>(&impl); memOp) {
    return memOp->readData();
  }
  return std::get<MemoryPortsImpl>(impl).readData;
}

std::optional<Value> MemoryInterface::readEnOpt() {
  if (auto *memOp = std::get_if<calyx::MemoryOp>(&impl); memOp) {
    return std::nullopt;
  }

  if (auto *memOp = std::get_if<calyx::SeqMemoryOp>(&impl); memOp) {
    return memOp->readEn();
  }
  return std::get<MemoryPortsImpl>(impl).readEn;
}

std::optional<Value> MemoryInterface::readDoneOpt() {
  if (auto *memOp = std::get_if<calyx::MemoryOp>(&impl); memOp) {
    return std::nullopt;
  }

  if (auto *memOp = std::get_if<calyx::SeqMemoryOp>(&impl); memOp) {
    return memOp->readDone();
  }
  return std::get<MemoryPortsImpl>(impl).readDone;
}
std::optional<Value> MemoryInterface::writeDataOpt() {
  if (auto *memOp = std::get_if<calyx::MemoryOp>(&impl); memOp) {
    return memOp->writeData();
  }

  if (auto *memOp = std::get_if<calyx::SeqMemoryOp>(&impl); memOp) {
    return memOp->writeData();
  }
  return std::get<MemoryPortsImpl>(impl).writeData;
}

std::optional<Value> MemoryInterface::writeEnOpt() {
  if (auto *memOp = std::get_if<calyx::MemoryOp>(&impl); memOp) {
    return memOp->writeEn();
  }

  if (auto *memOp = std::get_if<calyx::SeqMemoryOp>(&impl); memOp) {
    return memOp->writeEn();
  }
  return std::get<MemoryPortsImpl>(impl).writeEn;
}

std::optional<Value> MemoryInterface::writeDoneOpt() {
  if (auto *memOp = std::get_if<calyx::MemoryOp>(&impl); memOp) {
    return memOp->done();
  }

  if (auto *memOp = std::get_if<calyx::SeqMemoryOp>(&impl); memOp) {
    return memOp->writeDone();
  }
  return std::get<MemoryPortsImpl>(impl).writeDone;
}

ValueRange MemoryInterface::addrPorts() {
  if (auto *memOp = std::get_if<calyx::MemoryOp>(&impl); memOp) {
    return memOp->addrPorts();
  }

  if (auto *memOp = std::get_if<calyx::SeqMemoryOp>(&impl); memOp) {
    return memOp->addrPorts();
  }
  return std::get<MemoryPortsImpl>(impl).addrPorts;
}

//===----------------------------------------------------------------------===//
// BasicLoopInterface
//===----------------------------------------------------------------------===//

BasicLoopInterface::~BasicLoopInterface() = default;

//===----------------------------------------------------------------------===//
// ComponentLoweringStateInterface
//===----------------------------------------------------------------------===//

ComponentLoweringStateInterface::ComponentLoweringStateInterface(
    calyx::ComponentOp component)
    : component(component) {}

ComponentLoweringStateInterface::~ComponentLoweringStateInterface() = default;

calyx::ComponentOp ComponentLoweringStateInterface::getComponentOp() {
  return component;
}

void ComponentLoweringStateInterface::addBlockArgReg(Block *block,
                                                     calyx::RegisterOp reg,
                                                     unsigned idx) {
  assert(blockArgRegs[block].count(idx) == 0);
  assert(idx < block->getArguments().size());
  blockArgRegs[block][idx] = reg;
}

const DenseMap<unsigned, calyx::RegisterOp> &
ComponentLoweringStateInterface::getBlockArgRegs(Block *block) {
  return blockArgRegs[block];
}

void ComponentLoweringStateInterface::addBlockArgGroup(Block *from, Block *to,
                                                       calyx::GroupOp grp) {
  blockArgGroups[from][to].push_back(grp);
}

ArrayRef<calyx::GroupOp>
ComponentLoweringStateInterface::getBlockArgGroups(Block *from, Block *to) {
  return blockArgGroups[from][to];
}

std::string ComponentLoweringStateInterface::getUniqueName(StringRef prefix) {
  std::string prefixStr = prefix.str();
  unsigned idx = prefixIdMap[prefixStr];
  ++prefixIdMap[prefixStr];
  return (prefix + "_" + std::to_string(idx)).str();
}

StringRef ComponentLoweringStateInterface::getUniqueName(Operation *op) {
  auto it = opNames.find(op);
  assert(it != opNames.end() && "A unique name should have been set for op");
  return it->second;
}

void ComponentLoweringStateInterface::setUniqueName(Operation *op,
                                                    StringRef prefix) {
  assert(opNames.find(op) == opNames.end() &&
         "A unique name was already set for op");
  opNames[op] = getUniqueName(prefix);
}

void ComponentLoweringStateInterface::registerEvaluatingGroup(
    Value v, calyx::GroupInterface group) {
  valueGroupAssigns[v] = group;
}

void ComponentLoweringStateInterface::addReturnReg(calyx::RegisterOp reg,
                                                   unsigned idx) {
  assert(returnRegs.count(idx) == 0 &&
         "A register was already registered for this index");
  returnRegs[idx] = reg;
}

calyx::RegisterOp ComponentLoweringStateInterface::getReturnReg(unsigned idx) {
  assert(returnRegs.count(idx) && "No register registered for index!");
  return returnRegs[idx];
}

void ComponentLoweringStateInterface::registerMemoryInterface(
    Value memref, const calyx::MemoryInterface &memoryInterface) {
  assert(memref.getType().isa<MemRefType>());
  assert(memories.find(memref) == memories.end() &&
         "Memory already registered for memref");
  memories[memref] = memoryInterface;
}

calyx::MemoryInterface
ComponentLoweringStateInterface::getMemoryInterface(Value memref) {
  assert(memref.getType().isa<MemRefType>());
  auto it = memories.find(memref);
  assert(it != memories.end() && "No memory registered for memref");
  return it->second;
}

std::optional<calyx::MemoryInterface>
ComponentLoweringStateInterface::isInputPortOfMemory(Value v) {
  for (auto &memIf : memories) {
    auto &mem = memIf.getSecond();
    if (mem.writeEn() == v || mem.writeData() == v ||
        llvm::any_of(mem.addrPorts(), [=](Value port) { return port == v; }))
      return {mem};
  }
  return {};
}

void ComponentLoweringStateInterface::setFuncOpResultMapping(
    const DenseMap<unsigned, unsigned> &mapping) {
  funcOpResultMapping = mapping;
}

unsigned ComponentLoweringStateInterface::getFuncOpResultMapping(
    unsigned funcReturnIdx) {
  auto it = funcOpResultMapping.find(funcReturnIdx);
  assert(it != funcOpResultMapping.end() &&
         "No component return port index recorded for the requested function "
         "return index");
  return it->second;
}

//===----------------------------------------------------------------------===//
// CalyxLoweringState
//===----------------------------------------------------------------------===//

CalyxLoweringState::CalyxLoweringState(mlir::ModuleOp module,
                                       StringRef topLevelFunction)
    : topLevelFunction(topLevelFunction), module(module) {}

mlir::ModuleOp CalyxLoweringState::getModule() {
  assert(module.getOperation() != nullptr);
  return module;
}

StringRef CalyxLoweringState::getTopLevelFunction() const {
  return topLevelFunction;
}

std::string CalyxLoweringState::blockName(Block *b) {
  std::string blockName = irName(*b);
  blockName.erase(std::remove(blockName.begin(), blockName.end(), '^'),
                  blockName.end());
  return blockName;
}

//===----------------------------------------------------------------------===//
// ModuleOpConversion
//===----------------------------------------------------------------------===//

/// Helper to update the top-level ModuleOp to set the entrypoing function.
LogicalResult applyModuleOpConversion(mlir::ModuleOp moduleOp,
                                      StringRef topLevelFunction) {

  if (moduleOp->hasAttr("calyx.entrypoint"))
    return failure();

  moduleOp->setAttr("calyx.entrypoint",
                    StringAttr::get(moduleOp.getContext(), topLevelFunction));
  return success();
}

//===----------------------------------------------------------------------===//
// Partial lowering patterns
//===----------------------------------------------------------------------===//

FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern(
    MLIRContext *context, LogicalResult &resRef,
    PatternApplicationState &patternState,
    DenseMap<mlir::func::FuncOp, calyx::ComponentOp> &map,
    calyx::CalyxLoweringState &state)
    : PartialLoweringPattern(context, resRef, patternState),
      functionMapping(map), calyxLoweringState(state) {}

LogicalResult
FuncOpPartialLoweringPattern::partiallyLower(mlir::func::FuncOp funcOp,
                                             PatternRewriter &rewriter) const {
  // Initialize the component op references if a calyx::ComponentOp has been
  // created for the matched funcOp.
  if (auto it = functionMapping.find(funcOp); it != functionMapping.end()) {
    componentOp = it->second;
    componentLoweringState =
        calyxLoweringState.getState<ComponentLoweringStateInterface>(
            componentOp);
  }

  return partiallyLowerFuncToComp(funcOp, rewriter);
}

calyx::ComponentOp FuncOpPartialLoweringPattern::getComponent() const {
  assert(componentOp &&
         "Component operation should be set during pattern construction");
  return componentOp;
}

CalyxLoweringState &FuncOpPartialLoweringPattern::loweringState() const {
  return calyxLoweringState;
}

//===----------------------------------------------------------------------===//
// ConvertIndexTypes
//===----------------------------------------------------------------------===//

LogicalResult
ConvertIndexTypes::partiallyLowerFuncToComp(mlir::func::FuncOp funcOp,
                                            PatternRewriter &rewriter) const {
  funcOp.walk([&](Block *block) {
    for (Value arg : block->getArguments())
      arg.setType(calyx::convIndexType(rewriter, arg.getType()));
  });

  funcOp.walk([&](Operation *op) {
    for (Value result : op->getResults()) {
      Type resType = result.getType();
      if (!resType.isIndex())
        continue;

      result.setType(calyx::convIndexType(rewriter, resType));
      auto constant = dyn_cast<mlir::arith::ConstantOp>(op);
      if (!constant)
        continue;

      APInt value;
      calyx::matchConstantOp(constant, value);
      rewriter.setInsertionPoint(constant);
      rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(
          constant, rewriter.getI32IntegerAttr(value.getSExtValue()));
    }
  });
  return success();
}

//===----------------------------------------------------------------------===//
// NonTerminatingGroupDonePattern
//===----------------------------------------------------------------------===//

LogicalResult
NonTerminatingGroupDonePattern::matchAndRewrite(calyx::GroupDoneOp groupDoneOp,
                                                PatternRewriter &) const {
  Block *block = groupDoneOp->getBlock();
  if (&block->back() == groupDoneOp)
    return failure();

  groupDoneOp->moveBefore(groupDoneOp->getBlock(),
                          groupDoneOp->getBlock()->end());
  return success();
}

//===----------------------------------------------------------------------===//
// MultipleGroupDonePattern
//===----------------------------------------------------------------------===//

LogicalResult
MultipleGroupDonePattern::matchAndRewrite(calyx::GroupOp groupOp,
                                          PatternRewriter &rewriter) const {
  auto groupDoneOps = SmallVector<calyx::GroupDoneOp>(
      groupOp.getBodyBlock()->getOps<calyx::GroupDoneOp>());

  if (groupDoneOps.size() <= 1)
    return failure();

  /// 'and' all of the calyx::GroupDoneOp's.
  rewriter.setInsertionPointToEnd(groupDoneOps[0]->getBlock());
  SmallVector<Value> doneOpSrcs;
  llvm::transform(groupDoneOps, std::back_inserter(doneOpSrcs),
                  [](calyx::GroupDoneOp op) { return op.getSrc(); });
  Value allDone = rewriter.create<comb::AndOp>(groupDoneOps.front().getLoc(),
                                               doneOpSrcs, false);

  /// Create a group done op with the complex expression as a guard.
  rewriter.create<calyx::GroupDoneOp>(
      groupOp.getLoc(),
      rewriter.create<hw::ConstantOp>(groupOp.getLoc(), APInt(1, 1)), allDone);
  for (auto groupDoneOp : groupDoneOps)
    rewriter.eraseOp(groupDoneOp);

  return success();
}

//===----------------------------------------------------------------------===//
// EliminateUnusedCombGroups
//===----------------------------------------------------------------------===//

LogicalResult
EliminateUnusedCombGroups::matchAndRewrite(calyx::CombGroupOp combGroupOp,
                                           PatternRewriter &rewriter) const {
  auto control =
      combGroupOp->getParentOfType<calyx::ComponentOp>().getControlOp();
  if (!SymbolTable::symbolKnownUseEmpty(combGroupOp.getSymNameAttr(), control))
    return failure();

  rewriter.eraseOp(combGroupOp);
  return success();
}

//===----------------------------------------------------------------------===//
// InlineCombGroups
//===----------------------------------------------------------------------===//

InlineCombGroups::InlineCombGroups(MLIRContext *context, LogicalResult &resRef,
                                   PatternApplicationState &patternState,
                                   calyx::CalyxLoweringState &cls)
    : PartialLoweringPattern(context, resRef, patternState), cls(cls) {}

LogicalResult
InlineCombGroups::partiallyLower(calyx::GroupInterface originGroup,
                                 PatternRewriter &rewriter) const {
  auto component = originGroup->getParentOfType<calyx::ComponentOp>();
  ComponentLoweringStateInterface *state = cls.getState(component);

  // Filter groups which are not part of the control schedule.
  if (SymbolTable::symbolKnownUseEmpty(originGroup.symName(),
                                       component.getControlOp()))
    return success();

  // Maintain a set of the groups which we've inlined so far. The group
  // itself is implicitly inlined.
  llvm::SmallSetVector<Operation *, 8> inlinedGroups;
  inlinedGroups.insert(originGroup);

  // Starting from the matched originGroup, we traverse use-def chains of
  // combinational logic, and inline assignments from the defining
  // combinational groups.
  recurseInlineCombGroups(rewriter, *state, inlinedGroups, originGroup,
                          originGroup,
                          /*doInline=*/false);
  return success();
}

void InlineCombGroups::recurseInlineCombGroups(
    PatternRewriter &rewriter, ComponentLoweringStateInterface &state,
    llvm::SmallSetVector<Operation *, 8> &inlinedGroups,
    calyx::GroupInterface originGroup, calyx::GroupInterface recGroup,
    bool doInline) const {
  inlinedGroups.insert(recGroup);
  for (auto assignOp : recGroup.getBody()->getOps<calyx::AssignOp>()) {
    if (doInline) {
      /// Inline the assignment into the originGroup.
      auto *clonedAssignOp = rewriter.clone(*assignOp.getOperation());
      clonedAssignOp->moveBefore(originGroup.getBody(),
                                 originGroup.getBody()->end());
    }
    Value src = assignOp.getSrc();

    // Things which stop recursive inlining (or in other words, what
    // breaks combinational paths).
    // - Component inputs
    // - Register and memory reads
    // - Constant ops (constant ops are not evaluated by any group)
    // - Multiplication pipelines are sequential.
    // - 'While' return values (these are registers, however, 'while'
    //   return values have at the current point of conversion not yet
    //   been rewritten to their register outputs, see comment in
    //   LateSSAReplacement)
    if (src.isa<BlockArgument>() ||
        isa<calyx::RegisterOp, calyx::MemoryOp, calyx::SeqMemoryOp,
            hw::ConstantOp, mlir::arith::ConstantOp, calyx::MultPipeLibOp,
            calyx::DivUPipeLibOp, calyx::DivSPipeLibOp, calyx::RemSPipeLibOp,
            calyx::RemUPipeLibOp, mlir::scf::WhileOp>(src.getDefiningOp()))
      continue;

    auto srcCombGroup = dyn_cast<calyx::CombGroupOp>(
        state.getEvaluatingGroup(src).getOperation());
    if (!srcCombGroup)
      continue;
    if (inlinedGroups.count(srcCombGroup))
      continue;

    recurseInlineCombGroups(rewriter, state, inlinedGroups, originGroup,
                            srcCombGroup, /*doInline=*/true);
  }
}

//===----------------------------------------------------------------------===//
// RewriteMemoryAccesses
//===----------------------------------------------------------------------===//

LogicalResult
RewriteMemoryAccesses::partiallyLower(calyx::AssignOp assignOp,
                                      PatternRewriter &rewriter) const {
  auto *state = cls.getState(assignOp->getParentOfType<calyx::ComponentOp>());

  Value dest = assignOp.getDest();
  if (!state->isInputPortOfMemory(dest))
    return success();

  Value src = assignOp.getSrc();
  unsigned srcBits = src.getType().getIntOrFloatBitWidth();
  unsigned dstBits = dest.getType().getIntOrFloatBitWidth();
  if (srcBits == dstBits)
    return success();

  SmallVector<Type> types = {
      rewriter.getIntegerType(srcBits),
      rewriter.getIntegerType(dstBits),
  };
  mlir::Location loc = assignOp.getLoc();
  Operation *newOp;
  if (srcBits > dstBits)
    newOp =
        state->getNewLibraryOpInstance<calyx::SliceLibOp>(rewriter, loc, types);
  else
    newOp =
        state->getNewLibraryOpInstance<calyx::PadLibOp>(rewriter, loc, types);

  rewriter.setInsertionPoint(assignOp->getBlock(),
                             assignOp->getBlock()->begin());
  rewriter.create<calyx::AssignOp>(assignOp->getLoc(), newOp->getResult(0),
                                   src);
  assignOp.setOperand(1, newOp->getResult(1));

  return success();
}

//===----------------------------------------------------------------------===//
// BuildBasicBlockRegs
//===----------------------------------------------------------------------===//

LogicalResult
BuildBasicBlockRegs::partiallyLowerFuncToComp(mlir::func::FuncOp funcOp,
                                              PatternRewriter &rewriter) const {
  funcOp.walk([&](Block *block) {
    /// Do not register component input values.
    if (block == &block->getParent()->front())
      return;

    for (auto arg : enumerate(block->getArguments())) {
      Type argType = arg.value().getType();
      assert(argType.isa<IntegerType>() && "unsupported block argument type");
      unsigned width = argType.getIntOrFloatBitWidth();
      std::string index = std::to_string(arg.index());
      std::string name = loweringState().blockName(block) + "_arg" + index;
      auto reg = createRegister(arg.value().getLoc(), rewriter, getComponent(),
                                width, name);
      getState().addBlockArgReg(block, reg, arg.index());
      arg.value().replaceAllUsesWith(reg.getOut());
    }
  });
  return success();
}

//===----------------------------------------------------------------------===//
// BuildReturnRegs
//===----------------------------------------------------------------------===//

LogicalResult
BuildReturnRegs::partiallyLowerFuncToComp(mlir::func::FuncOp funcOp,
                                          PatternRewriter &rewriter) const {

  for (auto argType : enumerate(funcOp.getResultTypes())) {
    auto convArgType = calyx::convIndexType(rewriter, argType.value());
    assert(convArgType.isa<IntegerType>() && "unsupported return type");
    unsigned width = convArgType.getIntOrFloatBitWidth();
    std::string name = "ret_arg" + std::to_string(argType.index());
    auto reg =
        createRegister(funcOp.getLoc(), rewriter, getComponent(), width, name);
    getState().addReturnReg(reg, argType.index());

    rewriter.setInsertionPointToStart(
        getComponent().getWiresOp().getBodyBlock());
    rewriter.create<calyx::AssignOp>(
        funcOp->getLoc(),
        calyx::getComponentOutput(
            getComponent(), getState().getFuncOpResultMapping(argType.index())),
        reg.getOut());
  }
  return success();
}

} // namespace calyx
} // namespace circt

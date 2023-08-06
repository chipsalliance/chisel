//===- LowerCHIRRTL.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Transform CHIRRTL memory operations and memory ports into standard FIRRTL
// memory operations.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/CHIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace firrtl;
using namespace chirrtl;

namespace {
struct LowerCHIRRTLPass : public LowerCHIRRTLPassBase<LowerCHIRRTLPass>,
                          public CHIRRTLVisitor<LowerCHIRRTLPass>,
                          public FIRRTLVisitor<LowerCHIRRTLPass> {

  using FIRRTLVisitor<LowerCHIRRTLPass>::visitDecl;
  using FIRRTLVisitor<LowerCHIRRTLPass>::visitExpr;
  using FIRRTLVisitor<LowerCHIRRTLPass>::visitStmt;

  void visitCHIRRTL(CombMemOp op);
  void visitCHIRRTL(SeqMemOp op);
  void visitCHIRRTL(MemoryPortOp op);
  void visitCHIRRTL(MemoryDebugPortOp op);
  void visitCHIRRTL(MemoryPortAccessOp op);
  void visitExpr(SubaccessOp op);
  void visitExpr(SubfieldOp op);
  void visitExpr(SubindexOp op);
  void visitStmt(ConnectOp op);
  void visitStmt(StrictConnectOp op);
  void visitUnhandledOp(Operation *op);

  // Chain the CHIRRTL visitor to the FIRRTL visitor.
  void visitInvalidCHIRRTL(Operation *op) { dispatchVisitor(op); }
  void visitUnhandledCHIRRTL(Operation *op) { visitUnhandledOp(op); }

  /// Get a the constant 0.  This constant is inserted at the beginning of the
  /// module.
  Value getConst(unsigned c) {
    auto &value = constCache[c];
    if (!value) {
      auto module = getOperation();
      auto builder = OpBuilder::atBlockBegin(module.getBodyBlock());
      auto u1Type = UIntType::get(builder.getContext(), /*width*/ 1);
      value = builder.create<ConstantOp>(module.getLoc(), u1Type, APInt(1, c));
    }
    return value;
  }

  //// Clear out any stale data.
  void clear() {
    constCache.clear();
    invalidCache.clear();
    opsToDelete.clear();
    subfieldDirs.clear();
    rdataValues.clear();
    wdataValues.clear();
  }

  void emitInvalid(ImplicitLocOpBuilder &builder, Value value);

  MemDirAttr inferMemoryPortKind(MemoryPortOp memPort);

  void replaceMem(Operation *op, StringRef name, bool isSequential, RUWAttr ruw,
                  ArrayAttr annotations);

  template <typename OpType, typename... T>
  void cloneSubindexOpForMemory(OpType op, Value input, T... operands);

  void runOnOperation() override;

  /// Cached constants.
  DenseMap<unsigned, Value> constCache;
  DenseMap<Type, Value> invalidCache;

  /// List of operations to delete at the end of the pass.
  SmallVector<Operation *> opsToDelete;

  /// This tracks how the result of a subfield operation which is indexes a
  /// MemoryPortOp is used.  This is used to track if the subfield operation
  /// needs to be cloned to access a memories rdata or wdata.
  DenseMap<Operation *, MemDirAttr> subfieldDirs;

  /// This maps a subfield-like operation from a MemoryPortOp to a new subfield
  /// operation which can be used to read from the MemoryOp. This is used to
  /// update any operations to read from the new memory.
  DenseMap<Value, Value> rdataValues;

  /// This maps a subfield-like operation from a MemoryPortOp to a new subfield
  /// operation which can be used to write to the memory, the mask value which
  /// should be set to 1, and the corresponding wmode port of the memory which
  /// should be set to 1.  Not all memories have wmodes, so this field can
  /// be null. This is used to update operations to write to the new memory.
  struct WDataInfo {
    Value data;
    Value mask;
    Value mode;
  };
  DenseMap<Value, WDataInfo> wdataValues;
};
} // end anonymous namespace

/// Performs the callback for each leaf element of a value.  This will create
/// any subindex and subfield operations needed to access the leaf values of the
/// aggregate value.
static void forEachLeaf(ImplicitLocOpBuilder &builder, Value value,
                        llvm::function_ref<void(Value)> func) {
  auto type = value.getType();
  if (auto bundleType = type_dyn_cast<BundleType>(type)) {
    for (size_t i = 0, e = bundleType.getNumElements(); i < e; ++i)
      forEachLeaf(builder, builder.create<SubfieldOp>(value, i), func);
  } else if (auto vectorType = type_dyn_cast<FVectorType>(type)) {
    for (size_t i = 0, e = vectorType.getNumElements(); i != e; ++i)
      forEachLeaf(builder, builder.create<SubindexOp>(value, i), func);
  } else {
    func(value);
  }
}

/// Drive a value to all leafs of the input aggregate value. This only makes
/// sense when all leaf values have the same type, since the same value will be
/// connected to each leaf. This does not work for aggregates with flip types.
static void connectLeafsTo(ImplicitLocOpBuilder &builder, Value bundle,
                           Value value) {
  forEachLeaf(builder, bundle,
              [&](Value leaf) { emitConnect(builder, leaf, value); });
}

/// Connect each leaf of an aggregate type to invalid.  This does not support
/// aggregates with flip types.
void LowerCHIRRTLPass::emitInvalid(ImplicitLocOpBuilder &builder, Value value) {
  auto type = value.getType();
  auto &invalid = invalidCache[type];
  if (!invalid) {
    auto builder = OpBuilder::atBlockBegin(getOperation().getBodyBlock());
    invalid = builder.create<InvalidValueOp>(getOperation().getLoc(), type);
  }
  emitConnect(builder, value, invalid);
}

/// Converts a CHIRRTL memory port direction to a MemoryOp port type.  The
/// biggest difference is that there is no match for the Infer port type.
static MemOp::PortKind memDirAttrToPortKind(MemDirAttr direction) {
  switch (direction) {
  case MemDirAttr::Read:
    return MemOp::PortKind::Read;
  case MemDirAttr::Write:
    return MemOp::PortKind::Write;
  case MemDirAttr::ReadWrite:
    return MemOp::PortKind::ReadWrite;
  default:
    llvm_unreachable(
        "Unhandled MemDirAttr, was the port direction not inferred?");
  }
}

/// This function infers the memory direction of each CHIRRTL memory port. Each
/// memory port has an initial memory direction which is explicitly declared in
/// the MemoryPortOp, which is used as a starting point.  For example, if the
/// port is declared to be Write, but it is only ever read from, the port will
/// become a ReadWrite port.
///
/// When the memory port is eventually replaced with a memory, we will go from
/// having a single data value to having separate rdata and wdata values.  In
/// this function we record how the result of each data subfield operation is
/// used, so that later on we can make sure the SubfieldOp is cloned to index
/// into the correct rdata and wdata fields of the memory.
MemDirAttr LowerCHIRRTLPass::inferMemoryPortKind(MemoryPortOp memPort) {
  // This function does a depth-first walk of the use-lists of the memport
  // operation to look through subindex operations and find the places where it
  // is ultimately used.  At each node we record how the children ops are using
  // the result of the current operation.  When we are done visiting the current
  // operation we store how it is used into a global hashtable for later use.
  // This records how both the MemoryPort and Subfield operations are used.
  struct StackElement {
    StackElement(Value value, Value::use_iterator iterator, MemDirAttr mode)
        : value(value), iterator(iterator), mode(mode) {}
    Value value;
    Value::use_iterator iterator;
    MemDirAttr mode;
  };

  SmallVector<StackElement> stack;
  stack.emplace_back(memPort.getData(), memPort.getData().use_begin(),
                     memPort.getDirection());
  MemDirAttr mode = MemDirAttr::Infer;

  while (!stack.empty()) {
    auto *iter = &stack.back().iterator;
    auto end = stack.back().value.use_end();
    stack.back().mode |= mode;

    while (*iter != end) {
      auto &element = stack.back();
      auto &use = *(*iter);
      auto *user = use.getOwner();
      ++(*iter);
      if (isa<SubindexOp, SubfieldOp>(user)) {
        // We recurse into Subindex ops to find the leaf-uses.
        auto output = user->getResult(0);
        stack.emplace_back(output, output.use_begin(), MemDirAttr::Infer);
        mode = MemDirAttr::Infer;
        iter = &stack.back().iterator;
        end = output.use_end();
        continue;
      }
      if (auto subaccessOp = dyn_cast<SubaccessOp>(user)) {
        // Subaccess has two arguments, the vector and the index. If we are
        // using the memory port as an index, we can ignore it. If we are using
        // the memory as the vector, we need to recurse.
        auto input = subaccessOp.getInput();
        if (use.get() == input) {
          auto output = subaccessOp.getResult();
          stack.emplace_back(output, output.use_begin(), MemDirAttr::Infer);
          mode = MemDirAttr::Infer;
          iter = &stack.back().iterator;
          end = output.use_end();
          continue;
        }
        // Otherwise we are reading from a memory for the index.
        element.mode |= MemDirAttr::Read;
      } else if (auto connectOp = dyn_cast<ConnectOp>(user)) {
        if (use.get() == connectOp.getDest()) {
          element.mode |= MemDirAttr::Write;
        } else {
          element.mode |= MemDirAttr::Read;
        }
      } else if (auto connectOp = dyn_cast<StrictConnectOp>(user)) {
        if (use.get() == connectOp.getDest()) {
          element.mode |= MemDirAttr::Write;
        } else {
          element.mode |= MemDirAttr::Read;
        }
      } else {
        // Every other use of a memory is a read operation.
        element.mode |= MemDirAttr::Read;
      }
    }
    mode = stack.back().mode;

    // Store the direction of the current operation in the global map. This will
    // be used later to determine if this subaccess operation needs to be cloned
    // into rdata, wdata, and wmask.
    subfieldDirs[stack.back().value.getDefiningOp()] = mode;
    stack.pop_back();
  }

  return mode;
}

void LowerCHIRRTLPass::replaceMem(Operation *cmem, StringRef name,
                                  bool isSequential, RUWAttr ruw,
                                  ArrayAttr annotations) {
  assert(isa<CombMemOp>(cmem) || isa<SeqMemOp>(cmem));

  // We have several early breaks in this function, so we record the CHIRRTL
  // memory for deletion here.
  opsToDelete.push_back(cmem);
  ++numLoweredMems;

  auto cmemType = type_cast<CMemoryType>(cmem->getResult(0).getType());
  auto depth = cmemType.getNumElements();
  auto type = cmemType.getElementType();

  // Collect the information from each of the CMemoryPorts.
  struct PortInfo {
    StringAttr name;
    Type type;
    Attribute annotations;
    MemOp::PortKind portKind;
    Operation *cmemPort;
  };
  SmallVector<PortInfo, 4> ports;
  for (auto *user : cmem->getUsers()) {
    MemOp::PortKind portKind;
    StringAttr portName;
    ArrayAttr portAnnos;
    if (auto cmemoryPort = dyn_cast<MemoryPortOp>(user)) {
      // Infer the type of memory port we need to create.
      auto portDirection = inferMemoryPortKind(cmemoryPort);

      // If the memory port is never used, it will have the Infer type and
      // should just be deleted. TODO: this is mirroring SFC, but should we be
      // checking for annotations on the memory port before removing it?
      if (portDirection == MemDirAttr::Infer)
        continue;
      portKind = memDirAttrToPortKind(portDirection);
      portName = cmemoryPort.getNameAttr();
      portAnnos = cmemoryPort.getAnnotationsAttr();
    } else if (auto dPort = dyn_cast<MemoryDebugPortOp>(user)) {
      portKind = MemOp::PortKind::Debug;
      portName = dPort.getNameAttr();
      portAnnos = dPort.getAnnotationsAttr();
    } else {
      user->emitOpError("unhandled user of chirrtl memory");
      return;
    }

    // Add the new port.
    ports.push_back({portName, MemOp::getTypeForPort(depth, type, portKind),
                     portAnnos, portKind, user});
  }

  // If there are no valid memory ports, don't create a memory.
  if (ports.empty()) {
    ++numPortlessMems;
    return;
  }

  // Canonicalize the ports into alphabetical order.
  llvm::array_pod_sort(ports.begin(), ports.end(),
                       [](const PortInfo *lhs, const PortInfo *rhs) -> int {
                         return lhs->name.getValue().compare(
                             rhs->name.getValue());
                       });

  SmallVector<Attribute, 4> resultNames;
  SmallVector<Type, 4> resultTypes;
  SmallVector<Attribute, 4> portAnnotations;
  for (auto port : ports) {
    resultNames.push_back(port.name);
    resultTypes.push_back(port.type);
    portAnnotations.push_back(port.annotations);
  }

  // Write latency is always 1, while the read latency depends on the memory
  // type.
  auto readLatency = isSequential ? 1 : 0;
  auto writeLatency = 1;

  // Create the memory.
  ImplicitLocOpBuilder memBuilder(cmem->getLoc(), cmem);
  auto memory = memBuilder.create<MemOp>(
      resultTypes, readLatency, writeLatency, depth, ruw,
      memBuilder.getArrayAttr(resultNames), name,
      cmem->getAttrOfType<firrtl::NameKindEnumAttr>("nameKind").getValue(),
      annotations, memBuilder.getArrayAttr(portAnnotations), hw::InnerSymAttr(),
      cmem->getAttrOfType<firrtl::MemoryInitAttr>("init"), StringAttr());
  if (auto innerSym = cmem->getAttr("inner_sym"))
    memory->setAttr("inner_sym", innerSym);
  ++numCreatedMems;

  // Process each memory port, initializing the memory port and inferring when
  // to set the enable signal high.
  for (unsigned i = 0, e = memory.getNumResults(); i < e; ++i) {
    auto memoryPort = memory.getResult(i);
    auto portKind = ports[i].portKind;
    if (portKind == MemOp::PortKind::Debug) {
      rdataValues[ports[i].cmemPort->getResult(0)] = memoryPort;
      continue;
    }
    auto cmemoryPort = cast<MemoryPortOp>(ports[i].cmemPort);
    auto cmemoryPortAccess = cmemoryPort.getAccess();

    // Most fields on the newly created memory will be assigned an initial value
    // immediately following the memory decl, and then will be assigned a second
    // value at the location of the CHIRRTL memory port.

    // Initialization at the MemoryOp.
    ImplicitLocOpBuilder portBuilder(cmemoryPortAccess.getLoc(),
                                     cmemoryPortAccess);
    auto address = memBuilder.create<SubfieldOp>(memoryPort, "addr");
    emitInvalid(memBuilder, address);
    auto enable = memBuilder.create<SubfieldOp>(memoryPort, "en");
    emitConnect(memBuilder, enable, getConst(0));
    auto clock = memBuilder.create<SubfieldOp>(memoryPort, "clk");
    emitInvalid(memBuilder, clock);

    // Initialization at the MemoryPortOp.
    emitConnect(portBuilder, address, cmemoryPortAccess.getIndex());
    // Sequential+Read ports have a more complicated "enable inference".
    auto useEnableInference = isSequential && portKind == MemOp::PortKind::Read;
    auto *addressOp = cmemoryPortAccess.getIndex().getDefiningOp();
    // If the address value is not something with a "name", then we do not use
    // enable inference.
    useEnableInference &=
        !addressOp || isa<WireOp, NodeOp, RegOp, RegResetOp>(addressOp);

    // Most memory ports just tie their enable line to one.
    if (!useEnableInference)
      emitConnect(portBuilder, enable, getConst(1));

    emitConnect(portBuilder, clock, cmemoryPortAccess.getClock());

    if (portKind == MemOp::PortKind::Read) {
      // Store the read information for updating subfield ops.
      auto data = memBuilder.create<SubfieldOp>(memoryPort, "data");
      rdataValues[cmemoryPort.getData()] = data;
    } else if (portKind == MemOp::PortKind::Write) {
      // Initialization at the MemoryOp.
      auto data = memBuilder.create<SubfieldOp>(memoryPort, "data");
      emitInvalid(memBuilder, data);
      auto mask = memBuilder.create<SubfieldOp>(memoryPort, "mask");
      emitInvalid(memBuilder, mask);

      // Initialization at the MemoryPortOp.
      connectLeafsTo(portBuilder, mask, getConst(0));

      // Store the write information for updating subfield ops.
      wdataValues[cmemoryPort.getData()] = {data, mask, nullptr};
    } else if (portKind == MemOp::PortKind::ReadWrite) {
      // Initialization at the MemoryOp.
      auto rdata = memBuilder.create<SubfieldOp>(memoryPort, "rdata");
      auto wmode = memBuilder.create<SubfieldOp>(memoryPort, "wmode");
      emitConnect(memBuilder, wmode, getConst(0));
      auto wdata = memBuilder.create<SubfieldOp>(memoryPort, "wdata");
      emitInvalid(memBuilder, wdata);
      auto wmask = memBuilder.create<SubfieldOp>(memoryPort, "wmask");
      emitInvalid(memBuilder, wmask);

      // Initialization at the MemoryPortOp.
      connectLeafsTo(portBuilder, wmask, getConst(0));

      // Store the read and write information for updating subfield ops.
      wdataValues[cmemoryPort.getData()] = {wdata, wmask, wmode};
      rdataValues[cmemoryPort.getData()] = rdata;
    }

    // Sequential read only memory ports have "enable inference", which
    // detects when to set the enable high. All other memory ports set the
    // enable high when the memport is declared. This is higly questionable
    // logic that is easily defeated. This behaviour depends on the kind of
    // operation used as the memport index.
    if (useEnableInference) {
      auto *indexOp = cmemoryPortAccess.getIndex().getDefiningOp();
      bool success = false;
      if (!indexOp) {
        // TODO: SFC does not infer any enable when using a module port as the
        // address.  This seems like something that should be fixed sooner
        // rather than later.
      } else if (isa<WireOp, RegResetOp, RegOp>(indexOp)) {
        // If the address is a reference, then we set the enable whenever the
        // address is driven.

        // Find the uses of the address that write a value to it, ignoring the
        // ones driving an invalid value.
        auto drivers =
            make_filter_range(indexOp->getUsers(), [&](Operation *op) {
              if (auto connectOp = dyn_cast<ConnectOp>(op)) {
                if (cmemoryPortAccess.getIndex() == connectOp.getDest())
                  return !dyn_cast_or_null<InvalidValueOp>(
                      connectOp.getSrc().getDefiningOp());
              } else if (auto connectOp = dyn_cast<StrictConnectOp>(op)) {
                if (cmemoryPortAccess.getIndex() == connectOp.getDest())
                  return !dyn_cast_or_null<InvalidValueOp>(
                      connectOp.getSrc().getDefiningOp());
              }
              return false;
            });

        // At each location where we drive a value to the index, set the enable.
        for (auto *driver : drivers) {
          OpBuilder(driver).create<StrictConnectOp>(driver->getLoc(), enable,
                                                    getConst(1));
          success = true;
        }
      } else if (isa<NodeOp>(indexOp)) {
        // If using a Node for the address, then the we place the enable at the
        // Node op's
        OpBuilder(indexOp).create<StrictConnectOp>(indexOp->getLoc(), enable,
                                                   getConst(1));
        success = true;
      }

      // If we don't infer any enable points, it is almost always a user error.
      if (!success)
        cmemoryPort.emitWarning("memory port is never enabled");
    }
  }
}

void LowerCHIRRTLPass::visitCHIRRTL(CombMemOp combmem) {
  replaceMem(combmem, combmem.getName(), /*isSequential*/ false,
             RUWAttr::Undefined, combmem.getAnnotations());
}

void LowerCHIRRTLPass::visitCHIRRTL(SeqMemOp seqmem) {
  replaceMem(seqmem, seqmem.getName(), /*isSequential*/ true, seqmem.getRuw(),
             seqmem.getAnnotations());
}

void LowerCHIRRTLPass::visitCHIRRTL(MemoryPortOp memPort) {
  // The memory port is mostly handled while processing the memory.
  opsToDelete.push_back(memPort);
}

void LowerCHIRRTLPass::visitCHIRRTL(MemoryDebugPortOp memPort) {
  // The memory port is mostly handled while processing the memory.
  opsToDelete.push_back(memPort);
}

void LowerCHIRRTLPass::visitCHIRRTL(MemoryPortAccessOp memPortAccess) {
  // The memory port access is mostly handled while processing the memory.
  opsToDelete.push_back(memPortAccess);
}

void LowerCHIRRTLPass::visitStmt(ConnectOp connect) {
  // Check if we are writing to a memory and, if we are, replace the
  // destination.
  auto writeIt = wdataValues.find(connect.getDest());
  if (writeIt != wdataValues.end()) {
    auto writeData = writeIt->second;
    connect.getDestMutable().assign(writeData.data);
    // Assign the write mask.
    ImplicitLocOpBuilder builder(connect.getLoc(), connect);
    connectLeafsTo(builder, writeData.mask, getConst(1));
    // Only ReadWrite memories have a write mode.
    if (writeData.mode)
      emitConnect(builder, writeData.mode, getConst(1));
  }
  // Check if we are reading from a memory and, if we are, replace the
  // source.
  auto readIt = rdataValues.find(connect.getSrc());
  if (readIt != rdataValues.end()) {
    auto newSource = readIt->second;
    connect.getSrcMutable().assign(newSource);
  }
}

void LowerCHIRRTLPass::visitStmt(StrictConnectOp connect) {
  // Check if we are writing to a memory and, if we are, replace the
  // destination.
  auto writeIt = wdataValues.find(connect.getDest());
  if (writeIt != wdataValues.end()) {
    auto writeData = writeIt->second;
    connect.getDestMutable().assign(writeData.data);
    // Assign the write mask.
    ImplicitLocOpBuilder builder(connect.getLoc(), connect);
    connectLeafsTo(builder, writeData.mask, getConst(1));
    // Only ReadWrite memories have a write mode.
    if (writeData.mode)
      emitConnect(builder, writeData.mode, getConst(1));
  }
  // Check if we are reading from a memory and, if we are, replace the
  // source.
  auto readIt = rdataValues.find(connect.getSrc());
  if (readIt != rdataValues.end()) {
    auto newSource = readIt->second;
    connect.getSrcMutable().assign(newSource);
  }
}

/// This function will create clones of subaccess, subindex, and subfield
/// operations which are indexing a CHIRRTL memory ports that will index into
/// the new memory's data field.  If a subfield result is used to read from a
/// memory port, it will be cloned to read from the memory's rdata field.  If
/// the subfield is used to write to a memory port, it will be cloned twice to
/// write to both the wdata and wmask fields. Users of this subfield operation
/// will be redirected to the appropriate clone when they are visited.
template <typename OpType, typename... T>
void LowerCHIRRTLPass::cloneSubindexOpForMemory(OpType op, Value input,
                                                T... operands) {
  // If the subaccess operation has no direction recorded, then it does not
  // index a CHIRRTL memory and will be left alone.
  auto it = subfieldDirs.find(op);
  if (it == subfieldDirs.end()) {
    // The subaccess operation input could be a debug port of a CHIRRTL memory.
    // If it exists in the map, create the replacement operation for it.
    auto iter = rdataValues.find(input);
    if (iter != rdataValues.end()) {
      opsToDelete.push_back(op);
      ImplicitLocOpBuilder builder(op->getLoc(), op);
      rdataValues[op] = builder.create<OpType>(rdataValues[input], operands...);
    }
    return;
  }

  // All uses of this op will be updated to use the appropriate clone.  If the
  // recorded direction of this subfield is Infer, then the value is not
  // actually used to read or write from a memory port, and it will be just
  // removed.
  opsToDelete.push_back(op);

  auto direction = it->second;
  ImplicitLocOpBuilder builder(op->getLoc(), op);

  // If the subaccess operation is used to read from a memory port, we need to
  // clone it to read from the rdata field.
  if (direction == MemDirAttr::Read || direction == MemDirAttr::ReadWrite) {
    rdataValues[op] = builder.create<OpType>(rdataValues[input], operands...);
  }

  // If the subaccess operation is used to write to the memory, we need to clone
  // it to write to the wdata and the wmask fields.
  if (direction == MemDirAttr::Write || direction == MemDirAttr::ReadWrite) {
    auto writeData = wdataValues[input];
    auto write = builder.create<OpType>(writeData.data, operands...);
    auto mask = builder.create<OpType>(writeData.mask, operands...);
    wdataValues[op] = {write, mask, writeData.mode};
  }
}

void LowerCHIRRTLPass::visitExpr(SubaccessOp subaccess) {
  // Check if the subaccess reads from a memory for
  // the index.
  auto readIt = rdataValues.find(subaccess.getIndex());
  if (readIt != rdataValues.end()) {
    subaccess.getIndexMutable().assign(readIt->second);
  }
  // Handle it like normal.
  cloneSubindexOpForMemory(subaccess, subaccess.getInput(),
                           subaccess.getIndex());
}

void LowerCHIRRTLPass::visitExpr(SubfieldOp subfield) {
  cloneSubindexOpForMemory<SubfieldOp>(subfield, subfield.getInput(),
                                       subfield.getFieldIndex());
}

void LowerCHIRRTLPass::visitExpr(SubindexOp subindex) {
  cloneSubindexOpForMemory<SubindexOp>(subindex, subindex.getInput(),
                                       subindex.getIndex());
}

void LowerCHIRRTLPass::visitUnhandledOp(Operation *op) {
  // For every operand, check if it is reading from a memory port and
  // replace it with a read from the new memory.
  for (auto &operand : op->getOpOperands()) {
    auto it = rdataValues.find(operand.get());
    if (it != rdataValues.end()) {
      operand.set(it->second);
    }
  }
}

void LowerCHIRRTLPass::runOnOperation() {
  // Walk the entire body of the module and dispatch the visitor on each
  // function.  This will replace all CHIRRTL memories and ports, and update all
  // uses.
  getOperation().getBodyBlock()->walk(
      [&](Operation *op) { dispatchCHIRRTLVisitor(op); });

  // If there are no operations to delete, then we didn't find any CHIRRTL
  // memories.
  if (opsToDelete.empty())
    markAllAnalysesPreserved();

  // Remove the old memories and their ports.
  while (!opsToDelete.empty())
    opsToDelete.pop_back_val()->erase();

  // Clear out any cached data.
  clear();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createLowerCHIRRTLPass() {
  return std::make_unique<LowerCHIRRTLPass>();
}

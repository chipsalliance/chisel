//===- Simulation.cpp - Handshake MLIR Operations -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains functions used to execute a restricted form of the
// standard dialect, and the handshake dialect.
//
//===----------------------------------------------------------------------===//

#include <list>

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/Simulation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "runner"

#define INDEX_WIDTH 32

STATISTIC(instructionsExecuted, "Instructions Executed");
STATISTIC(simulatedTime, "Simulated Time");

using namespace llvm;
using namespace mlir;

namespace circt {
namespace handshake {

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

template <typename T>
static void fatalValueError(StringRef reason, T &value) {
  std::string err;
  llvm::raw_string_ostream os(err);
  os << reason << " ('";
  // Explicitly use ::print instead of << due to possibl operator resolution
  // error between i.e., mlir::Operation::<< and operator<<(OStream &&OS, const
  // T &Value)
  value.print(os);
  os << "')\n";
  llvm::report_fatal_error(err.c_str());
}

void debugArg(const std::string &head, mlir::Value op, const APInt &value,
              double time) {
  LLVM_DEBUG(dbgs() << "  " << head << ":  " << op << " = " << value
                    << " (APInt<" << value.getBitWidth() << ">) @" << time
                    << "\n");
}

void debugArg(const std::string &head, mlir::Value op, const APFloat &value,
              double time) {
  LLVM_DEBUG(dbgs() << "  " << head << ":  " << op << " = ";
             value.print(dbgs()); dbgs() << " ("
                                         << "float"
                                         << ") @" << time << "\n");
}

void debugArg(const std::string &head, mlir::Value op, const Any &value,
              double time) {
  if (auto *val = any_cast<APInt>(&value)) {
    debugArg(head, op, *val, time);
  } else if (auto *val = any_cast<APFloat>(&value)) {
    debugArg(head, op, val, time);
  } else if (auto *val = any_cast<unsigned>(&value)) {
    // Represents an allocated buffer.
    LLVM_DEBUG(dbgs() << "  " << head << ":  " << op << " = Buffer " << *val
                      << "\n");
  } else {
    llvm_unreachable("unknown type");
  }
}

Any readValueWithType(mlir::Type type, std::stringstream &arg) {
  if (type.isIndex()) {
    int64_t x;
    arg >> x;
    int64_t width = INDEX_WIDTH;
    APInt aparg(width, x);
    return aparg;
  } else if (type.isa<mlir::IntegerType>()) {
    int64_t x;
    arg >> x;
    int64_t width = type.getIntOrFloatBitWidth();
    APInt aparg(width, x);
    return aparg;
  } else if (type.isF32()) {
    float x;
    arg >> x;
    APFloat aparg(x);
    return aparg;
  } else if (type.isF64()) {
    double x;
    arg >> x;
    APFloat aparg(x);
    return aparg;
  } else if (auto tupleType = type.dyn_cast<TupleType>()) {
    char tmp;
    arg >> tmp;
    assert(tmp == '(' && "tuple should start with '('");
    std::vector<Any> values;
    unsigned size = tupleType.getTypes().size();
    values.reserve(size);
    // Parse element by element
    for (unsigned i = 0; i < size; ++i) {
      values.push_back(readValueWithType(tupleType.getType(i), arg));
      // Consumes either the ',' or the ')'
      arg >> tmp;
    }
    assert(tmp == ')' && "tuple should end with ')'");
    assert(
        values.size() == tupleType.getTypes().size() &&
        "expected the number of tuple elements to match with the tuple type");
    return values;
  } else {
    assert(false && "unknown argument type!");
    return {};
  }
}

Any readValueWithType(mlir::Type type, std::string in) {
  std::stringstream stream(in);
  return readValueWithType(type, stream);
}

void printAnyValueWithType(llvm::raw_ostream &out, mlir::Type type,
                           Any &value) {
  if (type.isa<mlir::IntegerType>() || type.isa<mlir::IndexType>()) {
    out << any_cast<APInt>(value).getSExtValue();
  } else if (type.isa<mlir::FloatType>()) {
    out << any_cast<APFloat>(value).convertToDouble();
  } else if (type.isa<mlir::NoneType>()) {
    out << "none";
  } else if (auto tupleType = type.dyn_cast<mlir::TupleType>()) {
    auto values = any_cast<std::vector<llvm::Any>>(value);
    out << "(";
    llvm::interleaveComma(llvm::zip(tupleType.getTypes(), values), out,
                          [&](auto pair) {
                            auto [type, value] = pair;
                            return printAnyValueWithType(out, type, value);
                          });
    out << ")";
  } else {
    llvm_unreachable("Unknown result type!");
  }
}

void scheduleIfNeeded(std::list<mlir::Operation *> &readyList,
                      llvm::DenseMap<mlir::Value, Any> & /*valueMap*/,
                      mlir::Operation *op) {
  if (std::find(readyList.begin(), readyList.end(), op) == readyList.end()) {
    readyList.push_back(op);
  }
}
void scheduleUses(std::list<mlir::Operation *> &readyList,
                  llvm::DenseMap<mlir::Value, Any> &valueMap,
                  mlir::Value value) {
  for (auto &use : value.getUses()) {
    scheduleIfNeeded(readyList, valueMap, use.getOwner());
  }
}

// Allocate a new matrix with dimensions given by the type, in the
// given store.  Return the pseudo-pointer to the new matrix in the
// store (i.e. the first dimension index).
unsigned allocateMemRef(mlir::MemRefType type, std::vector<Any> &in,
                        std::vector<std::vector<Any>> &store,
                        std::vector<double> &storeTimes) {
  ArrayRef<int64_t> shape = type.getShape();
  int64_t allocationSize = 1;
  unsigned count = 0;
  for (int64_t dim : shape) {
    if (dim > 0)
      allocationSize *= dim;
    else {
      assert(count < in.size());
      allocationSize *= any_cast<APInt>(in[count++]).getSExtValue();
    }
  }
  unsigned ptr = store.size();
  store.resize(ptr + 1);
  storeTimes.resize(ptr + 1);
  store[ptr].resize(allocationSize);
  storeTimes[ptr] = 0.0;
  mlir::Type elementType = type.getElementType();
  int64_t width = elementType.getIntOrFloatBitWidth();
  for (int i = 0; i < allocationSize; ++i) {
    if (elementType.isa<mlir::IntegerType>()) {
      store[ptr][i] = APInt(width, 0);
    } else if (elementType.isa<mlir::FloatType>()) {
      store[ptr][i] = APFloat(0.0);
    } else {
      fatalValueError("Unknown result type!\n", elementType);
    }
  }
  return ptr;
}

//===----------------------------------------------------------------------===//
// Handshake executer
//===----------------------------------------------------------------------===//

class HandshakeExecuter {
public:
  /// Entry point for mlir::func::FuncOp top-level functions
  HandshakeExecuter(mlir::func::FuncOp &toplevel,
                    llvm::DenseMap<mlir::Value, Any> &valueMap,
                    llvm::DenseMap<mlir::Value, double> &timeMap,
                    std::vector<Any> &results, std::vector<double> &resultTimes,
                    std::vector<std::vector<Any>> &store,
                    std::vector<double> &storeTimes);

  /// Entry point for handshake::FuncOp top-level functions
  HandshakeExecuter(handshake::FuncOp &func,
                    llvm::DenseMap<mlir::Value, Any> &valueMap,
                    llvm::DenseMap<mlir::Value, double> &timeMap,
                    std::vector<Any> &results, std::vector<double> &resultTimes,
                    std::vector<std::vector<Any>> &store,
                    std::vector<double> &storeTimes,
                    mlir::OwningOpRef<mlir::ModuleOp> &module);

  bool succeeded() const { return successFlag; }

private:
  /// Operation execution visitors
  LogicalResult execute(mlir::arith::ConstantIndexOp,
                        std::vector<Any> & /*inputs*/,
                        std::vector<Any> & /*outputs*/);
  LogicalResult execute(mlir::arith::ConstantIntOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::AddIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::XOrIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::AddFOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::CmpIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::CmpFOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::SubIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::SubFOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::MulIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::MulFOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::DivSIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::DivUIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::DivFOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::IndexCastOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::ExtSIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::arith::ExtUIOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(memref::LoadOp, std::vector<Any> &, std::vector<Any> &);
  LogicalResult execute(memref::StoreOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(memref::AllocOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::cf::BranchOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::cf::CondBranchOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(func::ReturnOp, std::vector<Any> &, std::vector<Any> &);
  LogicalResult execute(handshake::ReturnOp, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(mlir::CallOpInterface, std::vector<Any> &,
                        std::vector<Any> &);
  LogicalResult execute(handshake::InstanceOp, std::vector<Any> &,
                        std::vector<Any> &);

private:
  /// Execution context variables
  llvm::DenseMap<mlir::Value, Any> &valueMap;
  llvm::DenseMap<mlir::Value, double> &timeMap;
  std::vector<Any> &results;
  std::vector<double> &resultTimes;
  std::vector<std::vector<Any>> &store;
  std::vector<double> &storeTimes;
  double time;
  mlir::OwningOpRef<mlir::ModuleOp> *module = nullptr;

  /// Flag indicating whether execution was successful.
  bool successFlag = true;

  /// An iterator which walks over the instructions.
  mlir::Block::iterator instIter;
};

LogicalResult HandshakeExecuter::execute(mlir::arith::ConstantIndexOp op,
                                         std::vector<Any> &,
                                         std::vector<Any> &out) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>("value");
  out[0] = attr.getValue().sextOrTrunc(INDEX_WIDTH);
  return success();
}

LogicalResult HandshakeExecuter::execute(mlir::arith::ConstantIntOp op,
                                         std::vector<Any> &,
                                         std::vector<Any> &out) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>("value");
  out[0] = attr.getValue();
  return success();
}

LogicalResult HandshakeExecuter::execute(mlir::arith::XOrIOp,
                                         std::vector<Any> &in,
                                         std::vector<Any> &out) {
  out[0] = any_cast<APInt>(in[0]) ^ any_cast<APInt>(in[1]);
  return success();
}

LogicalResult HandshakeExecuter::execute(mlir::arith::AddIOp,
                                         std::vector<Any> &in,
                                         std::vector<Any> &out) {
  out[0] = any_cast<APInt>(in[0]) + any_cast<APInt>(in[1]);
  return success();
}

LogicalResult HandshakeExecuter::execute(mlir::arith::AddFOp,
                                         std::vector<Any> &in,
                                         std::vector<Any> &out) {
  out[0] = any_cast<APFloat>(in[0]) + any_cast<APFloat>(in[1]);
  return success();
}

LogicalResult HandshakeExecuter::execute(mlir::arith::CmpIOp op,
                                         std::vector<Any> &in,
                                         std::vector<Any> &out) {
  APInt in0 = any_cast<APInt>(in[0]);
  APInt in1 = any_cast<APInt>(in[1]);
  APInt out0(1, mlir::arith::applyCmpPredicate(op.getPredicate(), in0, in1));
  out[0] = out0;
  return success();
}

LogicalResult HandshakeExecuter::execute(mlir::arith::CmpFOp op,
                                         std::vector<Any> &in,
                                         std::vector<Any> &out) {
  APFloat in0 = any_cast<APFloat>(in[0]);
  APFloat in1 = any_cast<APFloat>(in[1]);
  APInt out0(1, mlir::arith::applyCmpPredicate(op.getPredicate(), in0, in1));
  out[0] = out0;
  return success();
}

LogicalResult HandshakeExecuter::execute(mlir::arith::SubIOp,
                                         std::vector<Any> &in,
                                         std::vector<Any> &out) {
  out[0] = any_cast<APInt>(in[0]) - any_cast<APInt>(in[1]);
  return success();
}

LogicalResult HandshakeExecuter::execute(mlir::arith::SubFOp,
                                         std::vector<Any> &in,
                                         std::vector<Any> &out) {
  out[0] = any_cast<APFloat>(in[0]) + any_cast<APFloat>(in[1]);
  return success();
}

LogicalResult HandshakeExecuter::execute(mlir::arith::MulIOp,
                                         std::vector<Any> &in,
                                         std::vector<Any> &out) {
  out[0] = any_cast<APInt>(in[0]) * any_cast<APInt>(in[1]);
  return success();
}

LogicalResult HandshakeExecuter::execute(mlir::arith::MulFOp,
                                         std::vector<Any> &in,
                                         std::vector<Any> &out) {
  out[0] = any_cast<APFloat>(in[0]) * any_cast<APFloat>(in[1]);
  return success();
}

LogicalResult HandshakeExecuter::execute(mlir::arith::DivSIOp op,
                                         std::vector<Any> &in,
                                         std::vector<Any> &out) {
  if (!any_cast<APInt>(in[1]).getZExtValue())
    return op.emitOpError() << "Division By Zero!";

  out[0] = any_cast<APInt>(in[0]).sdiv(any_cast<APInt>(in[1]));
  return success();
}

LogicalResult HandshakeExecuter::execute(mlir::arith::DivUIOp op,
                                         std::vector<Any> &in,
                                         std::vector<Any> &out) {
  if (!any_cast<APInt>(in[1]).getZExtValue())
    return op.emitOpError() << "Division By Zero!";
  out[0] = any_cast<APInt>(in[0]).udiv(any_cast<APInt>(in[1]));
  return success();
}

LogicalResult HandshakeExecuter::execute(mlir::arith::DivFOp,
                                         std::vector<Any> &in,
                                         std::vector<Any> &out) {
  out[0] = any_cast<APFloat>(in[0]) / any_cast<APFloat>(in[1]);
  return success();
}

LogicalResult HandshakeExecuter::execute(mlir::arith::IndexCastOp op,
                                         std::vector<Any> &in,
                                         std::vector<Any> &out) {
  Type outType = op.getOut().getType();
  APInt inValue = any_cast<APInt>(in[0]);
  APInt outValue;
  if (outType.isIndex())
    outValue =
        APInt(IndexType::kInternalStorageBitWidth, inValue.getZExtValue());
  else if (outType.isIntOrFloat())
    outValue = APInt(outType.getIntOrFloatBitWidth(), inValue.getZExtValue());
  else {
    return op.emitOpError() << "unhandled output type";
  }

  out[0] = outValue;
  return success();
}

LogicalResult HandshakeExecuter::execute(mlir::arith::ExtSIOp op,
                                         std::vector<Any> &in,
                                         std::vector<Any> &out) {
  int64_t width = op.getType().getIntOrFloatBitWidth();
  out[0] = any_cast<APInt>(in[0]).sext(width);
  return success();
}

LogicalResult HandshakeExecuter::execute(mlir::arith::ExtUIOp op,
                                         std::vector<Any> &in,
                                         std::vector<Any> &out) {
  int64_t width = op.getType().getIntOrFloatBitWidth();
  out[0] = any_cast<APInt>(in[0]).zext(width);
  return success();
}

LogicalResult HandshakeExecuter::execute(mlir::memref::LoadOp op,
                                         std::vector<Any> &in,
                                         std::vector<Any> &out) {
  ArrayRef<int64_t> shape = op.getMemRefType().getShape();
  unsigned address = 0;
  for (unsigned i = 0; i < shape.size(); ++i) {
    address = address * shape[i] + any_cast<APInt>(in[i + 1]).getZExtValue();
  }
  unsigned ptr = any_cast<unsigned>(in[0]);
  if (ptr >= store.size())
    return op.emitOpError()
           << "Unknown memory identified by pointer '" << ptr << "'";

  auto &ref = store[ptr];
  if (address >= ref.size())
    return op.emitOpError()
           << "Out-of-bounds access to memory '" << ptr << "'. Memory has "
           << ref.size() << " elements but requested element " << address;

  Any result = ref[address];
  out[0] = result;

  double storeTime = storeTimes[ptr];
  LLVM_DEBUG(dbgs() << "STORE: " << storeTime << "\n");
  time = std::max(time, storeTime);
  storeTimes[ptr] = time;
  return success();
}

LogicalResult HandshakeExecuter::execute(mlir::memref::StoreOp op,
                                         std::vector<Any> &in,
                                         std::vector<Any> &) {
  ArrayRef<int64_t> shape = op.getMemRefType().getShape();
  unsigned address = 0;
  for (unsigned i = 0; i < shape.size(); ++i) {
    address = address * shape[i] + any_cast<APInt>(in[i + 2]).getZExtValue();
  }
  unsigned ptr = any_cast<unsigned>(in[1]);
  if (ptr >= store.size())
    return op.emitOpError()
           << "Unknown memory identified by pointer '" << ptr << "'";
  auto &ref = store[ptr];
  if (address >= ref.size())
    return op.emitOpError()
           << "Out-of-bounds access to memory '" << ptr << "'. Memory has "
           << ref.size() << " elements but requested element " << address;
  ref[address] = in[0];

  double storeTime = storeTimes[ptr];
  LLVM_DEBUG(dbgs() << "STORE: " << storeTime << "\n");
  time = std::max(time, storeTime);
  storeTimes[ptr] = time;
  return success();
}

LogicalResult HandshakeExecuter::execute(mlir::memref::AllocOp op,
                                         std::vector<Any> &in,
                                         std::vector<Any> &out) {
  out[0] = allocateMemRef(op.getType(), in, store, storeTimes);
  unsigned ptr = any_cast<unsigned>(out[0]);
  storeTimes[ptr] = time;
  return success();
}

LogicalResult HandshakeExecuter::execute(mlir::cf::BranchOp branchOp,
                                         std::vector<Any> &in,
                                         std::vector<Any> &) {
  mlir::Block *dest = branchOp.getDest();
  for (auto out : enumerate(dest->getArguments())) {
    LLVM_DEBUG(debugArg("ARG", out.value(), in[out.index()], time));
    valueMap[out.value()] = in[out.index()];
    timeMap[out.value()] = time;
  }
  instIter = dest->begin();
  return success();
}

LogicalResult HandshakeExecuter::execute(mlir::cf::CondBranchOp condBranchOp,
                                         std::vector<Any> &in,
                                         std::vector<Any> &) {
  APInt condition = any_cast<APInt>(in[0]);
  mlir::Block *dest;
  std::vector<Any> inArgs;
  double time = 0.0;
  if (condition != 0) {
    dest = condBranchOp.getTrueDest();
    inArgs.resize(condBranchOp.getNumTrueOperands());
    for (auto in : enumerate(condBranchOp.getTrueOperands())) {
      inArgs[in.index()] = valueMap[in.value()];
      time = std::max(time, timeMap[in.value()]);
      LLVM_DEBUG(
          debugArg("IN", in.value(), inArgs[in.index()], timeMap[in.value()]));
    }
  } else {
    dest = condBranchOp.getFalseDest();
    inArgs.resize(condBranchOp.getNumFalseOperands());
    for (auto in : enumerate(condBranchOp.getFalseOperands())) {
      inArgs[in.index()] = valueMap[in.value()];
      time = std::max(time, timeMap[in.value()]);
      LLVM_DEBUG(
          debugArg("IN", in.value(), inArgs[in.index()], timeMap[in.value()]));
    }
  }
  for (auto out : enumerate(dest->getArguments())) {
    LLVM_DEBUG(debugArg("ARG", out.value(), inArgs[out.index()], time));
    valueMap[out.value()] = inArgs[out.index()];
    timeMap[out.value()] = time;
  }
  instIter = dest->begin();
  return success();
}

LogicalResult HandshakeExecuter::execute(func::ReturnOp op,
                                         std::vector<Any> &in,
                                         std::vector<Any> &) {
  for (unsigned i = 0; i < results.size(); ++i) {
    results[i] = in[i];
    resultTimes[i] = timeMap[op.getOperand(i)];
  }
  return success();
}

LogicalResult HandshakeExecuter::execute(handshake::ReturnOp op,
                                         std::vector<Any> &in,
                                         std::vector<Any> &) {
  for (unsigned i = 0; i < results.size(); ++i) {
    results[i] = in[i];
    resultTimes[i] = timeMap[op.getOperand(i)];
  }
  return success();
}

LogicalResult HandshakeExecuter::execute(mlir::CallOpInterface callOp,
                                         std::vector<Any> &in,
                                         std::vector<Any> &) {
  // implement function calls.
  auto op = callOp.getOperation();
  mlir::Operation *calledOp = callOp.resolveCallable();
  if (auto funcOp = dyn_cast<mlir::func::FuncOp>(calledOp)) {
    unsigned outputs = funcOp.getNumResults();
    llvm::DenseMap<mlir::Value, Any> newValueMap;
    llvm::DenseMap<mlir::Value, double> newTimeMap;
    std::vector<Any> results(outputs);
    std::vector<double> resultTimes(outputs);
    std::vector<std::vector<Any>> store;
    std::vector<double> storeTimes;
    mlir::Block &entryBlock = funcOp.getBody().front();
    mlir::Block::BlockArgListType blockArgs = entryBlock.getArguments();

    for (auto inIt : enumerate(in)) {
      newValueMap[blockArgs[inIt.index()]] = inIt.value();
      newTimeMap[blockArgs[inIt.index()]] =
          timeMap[op->getOperand(inIt.index())];
    }
    HandshakeExecuter(funcOp, newValueMap, newTimeMap, results, resultTimes,
                      store, storeTimes);
    for (auto out : enumerate(op->getResults())) {
      valueMap[out.value()] = results[out.index()];
      timeMap[out.value()] = resultTimes[out.index()];
    }
    ++instIter;
  } else
    return op->emitOpError() << "Callable was not a function";

  return success();
}

LogicalResult HandshakeExecuter::execute(handshake::InstanceOp instanceOp,
                                         std::vector<Any> &in,
                                         std::vector<Any> &out) {
  // Execute the instance op and create associations in the current
  // scope's value and time maps for the returned values.

  if (auto funcSym = instanceOp->getAttr("module").cast<SymbolRefAttr>()) {
    if (handshake::FuncOp func =
            (*module)->lookupSymbol<handshake::FuncOp>(funcSym)) {
      /// Prepare an InstanceOp for execution by creating a valueMap
      /// containing associations between the arguments provided to the
      /// intanceOp - available in the enclosing scope value map - and the
      /// argument SSA values within the called function of the InstanceOp.

      const unsigned nRealFuncOuts = func.getNumResults() - 1;
      mlir::Block &entryBlock = func.getBody().front();
      mlir::Block::BlockArgListType instanceBlockArgs =
          entryBlock.getArguments();

      // Create a new value map containing only the arguments of the
      // InstanceOp. This will be the value and time map for the callee scope of
      // the function pointed to by the InstanceOp.
      llvm::DenseMap<mlir::Value, Any> scopeValueMap;
      llvm::DenseMap<mlir::Value, double> scopeTimeMap;

      // Associate each input argument with the arguments of the called
      // function
      for (auto inIt : enumerate(in)) {
        scopeValueMap[instanceBlockArgs[inIt.index()]] = inIt.value();
        scopeTimeMap[instanceBlockArgs[inIt.index()]] =
            timeMap[instanceOp.getOperand(inIt.index())];
      }

      // ... and the implicit none argument
      APInt apnonearg(1, 0);
      scopeValueMap[instanceBlockArgs[instanceBlockArgs.size() - 1]] =
          apnonearg;
      std::vector<Any> nestedResults(nRealFuncOuts);
      std::vector<double> nestedResTimes(nRealFuncOuts);

      // Go execute!
      HandshakeExecuter(func, scopeValueMap, scopeTimeMap, nestedResults,
                        nestedResTimes, store, storeTimes, *module);

      // Place the output arguments in the caller scope.
      for (auto nestedRes : enumerate(nestedResults)) {
        out[nestedRes.index()] = nestedRes.value();
        valueMap[instanceOp.getResults()[nestedRes.index()]] =
            nestedRes.value();
        timeMap[instanceOp.getResults()[nestedRes.index()]] =
            nestedResTimes[nestedRes.index()];
      }
      // ... and the implicit none argument
      unsigned ctrlResultIdx = instanceOp.getNumResults() - 1;
      valueMap[instanceOp->getResult(ctrlResultIdx)] = apnonearg;
      out[ctrlResultIdx] = apnonearg;

      return success();
    } else {
      return instanceOp.emitOpError()
             << "Function '" << funcSym << "' not found in module";
    }
  } else
    return instanceOp.emitOpError()
           << "Missing 'module' attribute for InstanceOp";

  llvm_unreachable("Fatal error reached before this point");
}

enum ExecuteStrategy { Default = 1 << 0, Continue = 1 << 1, Return = 1 << 2 };

HandshakeExecuter::HandshakeExecuter(
    mlir::func::FuncOp &toplevel, llvm::DenseMap<mlir::Value, Any> &valueMap,
    llvm::DenseMap<mlir::Value, double> &timeMap, std::vector<Any> &results,
    std::vector<double> &resultTimes, std::vector<std::vector<Any>> &store,
    std::vector<double> &storeTimes)
    : valueMap(valueMap), timeMap(timeMap), results(results),
      resultTimes(resultTimes), store(store), storeTimes(storeTimes) {
  successFlag = true;
  mlir::Block &entryBlock = toplevel.getBody().front();
  instIter = entryBlock.begin();

  // Main executive loop.  Start at the first instruction of the entry
  // block.  Fetch and execute instructions until we hit a terminator.
  while (true) {
    mlir::Operation &op = *instIter;
    std::vector<Any> inValues(op.getNumOperands());
    std::vector<Any> outValues(op.getNumResults());
    LLVM_DEBUG(dbgs() << "OP:  " << op.getName() << "\n");
    time = 0.0;
    for (auto in : enumerate(op.getOperands())) {
      inValues[in.index()] = valueMap[in.value()];
      time = std::max(time, timeMap[in.value()]);
      LLVM_DEBUG(debugArg("IN", in.value(), inValues[in.index()],
                          timeMap[in.value()]));
    }

    unsigned strat = ExecuteStrategy::Default;
    auto res =
        llvm::TypeSwitch<Operation *, LogicalResult>(&op)
            .Case<mlir::arith::ConstantIndexOp, mlir::arith::ConstantIntOp,
                  mlir::arith::AddIOp, mlir::arith::AddFOp, mlir::arith::CmpIOp,
                  mlir::arith::CmpFOp, mlir::arith::SubIOp, mlir::arith::SubFOp,
                  mlir::arith::MulIOp, mlir::arith::MulFOp,
                  mlir::arith::DivSIOp, mlir::arith::DivUIOp,
                  mlir::arith::DivFOp, mlir::arith::IndexCastOp,
                  mlir::arith::ExtSIOp, mlir::arith::ExtUIOp, memref::AllocOp,
                  memref::LoadOp, memref::StoreOp>([&](auto op) {
              strat = ExecuteStrategy::Default;
              return execute(op, inValues, outValues);
            })
            .Case<mlir::cf::BranchOp, mlir::cf::CondBranchOp,
                  mlir::CallOpInterface>([&](auto op) {
              strat = ExecuteStrategy::Continue;
              return execute(op, inValues, outValues);
            })
            .Case<func::ReturnOp>([&](auto op) {
              strat = ExecuteStrategy::Return;
              return execute(op, inValues, outValues);
            })
            .Default([](auto op) {
              return op->emitOpError() << "Unknown operation!";
            });

    if (res.failed()) {
      successFlag = false;
      return;
    }

    if (strat & ExecuteStrategy::Continue)
      continue;

    if (strat & ExecuteStrategy::Return)
      return;

    for (auto out : enumerate(op.getResults())) {
      LLVM_DEBUG(debugArg("OUT", out.value(), outValues[out.index()], time));
      valueMap[out.value()] = outValues[out.index()];
      timeMap[out.value()] = time + 1;
    }
    ++instIter;
    ++instructionsExecuted;
  }
}

HandshakeExecuter::HandshakeExecuter(
    handshake::FuncOp &func, llvm::DenseMap<mlir::Value, Any> &valueMap,
    llvm::DenseMap<mlir::Value, double> &timeMap, std::vector<Any> &results,
    std::vector<double> &resultTimes, std::vector<std::vector<Any>> &store,
    std::vector<double> &storeTimes, mlir::OwningOpRef<mlir::ModuleOp> &module)
    : valueMap(valueMap), timeMap(timeMap), results(results),
      resultTimes(resultTimes), store(store), storeTimes(storeTimes),
      module(&module) {
  successFlag = true;
  mlir::Block &entryBlock = func.getBody().front();
  // The arguments of the entry block.
  mlir::Block::BlockArgListType blockArgs = entryBlock.getArguments();
  // A list of operations which might be ready to execute.
  std::list<mlir::Operation *> readyList;
  // A map of memory ops
  llvm::DenseMap<unsigned, unsigned> memoryMap;

  // Pre-allocate memory
  func.walk([&](Operation *op) {
    if (auto handshakeMemoryOp = dyn_cast<handshake::MemoryOpInterface>(op))
      if (!handshakeMemoryOp.allocateMemory(memoryMap, store, storeTimes))
        llvm_unreachable("Memory op does not have unique ID!\n");
  });

  // Initialize the value map for buffers with initial values.
  for (auto bufferOp : func.getOps<handshake::BufferOp>()) {
    if (bufferOp.getInitValues().has_value()) {
      auto initValues = bufferOp.getInitValueArray();
      assert(initValues.size() == 1 &&
             "Handshake-runner only supports buffer initialization with a "
             "single buffer value.");
      Value bufferRes = bufferOp.getResult();
      valueMap[bufferRes] = APInt(bufferRes.getType().getIntOrFloatBitWidth(),
                                  initValues.front());
      scheduleUses(readyList, valueMap, bufferRes);
    }
  }

  for (auto blockArg : blockArgs)
    scheduleUses(readyList, valueMap, blockArg);

#define EXTRA_DEBUG
  while (true) {
#ifdef EXTRA_DEBUG
    LLVM_DEBUG(
        for (auto t
             : readyList) { dbgs() << "READY: " << *t << "\n"; } dbgs()
            << "Live: " << valueMap.size() << "\n";
        for (auto t
             : valueMap) { debugArg("Value:", t.first, t.second, 0.0); });
#endif
    assert(readyList.size() > 0 &&
           "Expected some instruction to be ready for execution");
    mlir::Operation &op = *readyList.front();
    readyList.pop_front();

    // Execute handshake ops through ExecutableOpInterface
    if (auto handshakeOp = dyn_cast<handshake::ExecutableOpInterface>(op)) {
      std::vector<mlir::Value> scheduleList;
      if (!handshakeOp.tryExecute(valueMap, memoryMap, timeMap, store,
                                  scheduleList))
        readyList.push_back(&op);
      else {
        LLVM_DEBUG({
          dbgs() << "EXECUTED: " << op << "\n";
          for (auto out : op.getResults()) {
            auto valueIt = valueMap.find(out);
            if (valueIt != valueMap.end())
              debugArg("OUT", out, valueMap[out], time);
          }
        });
      }
      for (mlir::Value out : scheduleList)
        scheduleUses(readyList, valueMap, out);
      continue;
    }

    int64_t i = 0;
    std::vector<Any> inValues(op.getNumOperands());
    std::vector<Any> outValues(op.getNumResults());
    bool reschedule = false;
    LLVM_DEBUG(dbgs() << "OP: (" << op.getNumOperands() << "->"
                      << op.getNumResults() << ")" << op << "\n");
    time = 0;
    for (mlir::Value in : op.getOperands()) {
      if (valueMap.count(in) == 0) {
        reschedule = true;
        continue;
      }
      inValues[i] = valueMap[in];
      time = std::max(time, timeMap[in]);
      LLVM_DEBUG(debugArg("IN", in, inValues[i], timeMap[in]));
      ++i;
    }
    if (reschedule) {
      LLVM_DEBUG(dbgs() << "Rescheduling data...\n");
      readyList.push_back(&op);
      continue;
    }
    // Consume the inputs.
    for (mlir::Value in : op.getOperands())
      valueMap.erase(in);

    ExecuteStrategy strat = ExecuteStrategy::Default;
    LogicalResult res =
        llvm::TypeSwitch<Operation *, LogicalResult>(&op)
            .Case<mlir::arith::ConstantIndexOp, mlir::arith::ConstantIntOp,
                  mlir::arith::AddIOp, mlir::arith::AddFOp, mlir::arith::CmpIOp,
                  mlir::arith::CmpFOp, mlir::arith::SubIOp, mlir::arith::SubFOp,
                  mlir::arith::MulIOp, mlir::arith::MulFOp,
                  mlir::arith::DivSIOp, mlir::arith::DivUIOp,
                  mlir::arith::DivFOp, mlir::arith::IndexCastOp,
                  mlir::arith::ExtSIOp, mlir::arith::ExtUIOp,
                  mlir::arith::XOrIOp, handshake::InstanceOp>([&](auto op) {
              strat = ExecuteStrategy::Default;
              return execute(op, inValues, outValues);
            })
            .Case<handshake::ReturnOp>([&](auto op) {
              strat = ExecuteStrategy::Return;
              return execute(op, inValues, outValues);
            })
            .Default([](auto op) {
              return op->emitOpError() << "Unknown operation";
            });
    LLVM_DEBUG(dbgs() << "EXECUTED: " << op << "\n");

    if (res.failed()) {
      successFlag = false;
      return;
    }

    if (strat & ExecuteStrategy::Return)
      return;

    for (auto out : enumerate(op.getResults())) {
      LLVM_DEBUG(debugArg("OUT", out.value(), outValues[out.index()], time));
      assert(outValues[out.index()].has_value());
      valueMap[out.value()] = outValues[out.index()];
      timeMap[out.value()] = time + 1;
      scheduleUses(readyList, valueMap, out.value());
    }
    ++instructionsExecuted;
  }
}

//===----------------------------------------------------------------------===//
// Simulator entry point
//===----------------------------------------------------------------------===//

bool simulate(StringRef toplevelFunction, ArrayRef<std::string> inputArgs,
              mlir::OwningOpRef<mlir::ModuleOp> &module, mlir::MLIRContext &) {
  // The store associates each allocation in the program
  // (represented by a int) with a vector of values which can be
  // accessed by it.  Currently values are assumed to be an integer.
  std::vector<std::vector<Any>> store;
  std::vector<double> storeTimes;

  // The valueMap associates each SSA statement in the program
  // (represented by a Value*) with it's corresponding value.
  // Currently the value is assumed to be an integer.
  llvm::DenseMap<mlir::Value, Any> valueMap;

  // The timeMap associates each value with the time it was created.
  llvm::DenseMap<mlir::Value, double> timeMap;

  // We need three things in a function-type independent way.
  // The type signature of the function.
  mlir::FunctionType ftype;
  // The arguments of the entry block.
  mlir::Block::BlockArgListType blockArgs;

  // The number of inputs to the function in the IR.
  unsigned inputs;
  unsigned outputs;
  // The number of 'real' inputs.  This avoids the dummy input
  // associated with the handshake control logic for handshake
  // functions.
  unsigned realInputs;
  unsigned realOutputs;

  if (mlir::func::FuncOp toplevel =
          module->lookupSymbol<mlir::func::FuncOp>(toplevelFunction)) {
    ftype = toplevel.getFunctionType();
    mlir::Block &entryBlock = toplevel.getBody().front();
    blockArgs = entryBlock.getArguments();

    // Get the primary inputs of toplevel off the command line.
    inputs = toplevel.getNumArguments();
    realInputs = inputs;
    outputs = toplevel.getNumResults();
    realOutputs = outputs;
  } else if (handshake::FuncOp toplevel =
                 module->lookupSymbol<handshake::FuncOp>(toplevelFunction)) {
    ftype = toplevel.getFunctionType();
    mlir::Block &entryBlock = toplevel.getBody().front();
    blockArgs = entryBlock.getArguments();

    // Get the primary inputs of toplevel off the command line.
    inputs = toplevel.getNumArguments();
    realInputs = inputs - 1;
    outputs = toplevel.getNumResults();
    realOutputs = outputs - 1;
    if (inputs == 0) {
      errs() << "Function " << toplevelFunction << " is expected to have "
             << "at least one dummy argument.\n";
      return 1;
    }
    if (outputs == 0) {
      errs() << "Function " << toplevelFunction << " is expected to have "
             << "at least one dummy result.\n";
      return 1;
    }
    // Implicit none argument
    APInt apnonearg(1, 0);
    valueMap[blockArgs[blockArgs.size() - 1]] = apnonearg;
  } else
    llvm::report_fatal_error("Function '" + toplevelFunction +
                             "' not supported");

  if (inputArgs.size() != realInputs) {
    errs() << "Toplevel function " << toplevelFunction << " has " << realInputs
           << " actual arguments, but " << inputArgs.size()
           << " arguments were provided on the command line.\n";
    return 1;
  }

  for (unsigned i = 0; i < realInputs; ++i) {
    mlir::Type type = ftype.getInput(i);
    if (type.isa<mlir::MemRefType>()) {
      // We require this memref type to be fully specified.
      auto memreftype = type.dyn_cast<mlir::MemRefType>();
      std::vector<Any> nothing;
      std::string x;
      unsigned buffer = allocateMemRef(memreftype, nothing, store, storeTimes);
      valueMap[blockArgs[i]] = buffer;
      timeMap[blockArgs[i]] = 0.0;
      int64_t i = 0;
      std::stringstream arg(inputArgs[i]);
      while (!arg.eof()) {
        getline(arg, x, ',');
        store[buffer][i++] = readValueWithType(memreftype.getElementType(), x);
      }
    } else {
      Any value = readValueWithType(type, inputArgs[i]);
      valueMap[blockArgs[i]] = value;
      timeMap[blockArgs[i]] = 0.0;
    }
  }

  std::vector<Any> results(realOutputs);
  std::vector<double> resultTimes(realOutputs);
  bool succeeded = false;
  if (mlir::func::FuncOp toplevel =
          module->lookupSymbol<mlir::func::FuncOp>(toplevelFunction)) {
    succeeded = HandshakeExecuter(toplevel, valueMap, timeMap, results,
                                  resultTimes, store, storeTimes)
                    .succeeded();
  } else if (handshake::FuncOp toplevel =
                 module->lookupSymbol<handshake::FuncOp>(toplevelFunction)) {
    succeeded = HandshakeExecuter(toplevel, valueMap, timeMap, results,
                                  resultTimes, store, storeTimes, module)
                    .succeeded();
  }

  if (!succeeded)
    return 1;

  double time = 0.0;
  for (unsigned i = 0; i < results.size(); ++i) {
    mlir::Type t = ftype.getResult(i);
    printAnyValueWithType(outs(), t, results[i]);
    outs() << " ";
    time = std::max(resultTimes[i], time);
  }
  // Go back through the arguments and output any memrefs.
  for (unsigned i = 0; i < realInputs; ++i) {
    mlir::Type type = ftype.getInput(i);
    if (type.isa<mlir::MemRefType>()) {
      // We require this memref type to be fully specified.
      auto memreftype = type.dyn_cast<mlir::MemRefType>();
      unsigned buffer = any_cast<unsigned>(valueMap[blockArgs[i]]);
      auto elementType = memreftype.getElementType();
      for (int j = 0; j < memreftype.getNumElements(); ++j) {
        if (j != 0)
          outs() << ",";
        printAnyValueWithType(outs(), elementType, store[buffer][j]);
      }
      outs() << " ";
    }
  }
  outs() << "\n";

  simulatedTime += (int)time;

  return 0;
}

} // namespace handshake
} // namespace circt

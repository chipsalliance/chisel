//===- HandshakeExecutableOps.cpp - Handshake executable Operations -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of execution semantics for Handshake
// operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/Any.h"
#include "llvm/Support/Debug.h"

using namespace circt;
using namespace circt::handshake;

#define INDEX_WIDTH 32

// Convert ValueRange to vectors
static std::vector<mlir::Value> toVector(mlir::ValueRange range) {
  return std::vector<mlir::Value>(range.begin(), range.end());
}

// Returns whether the precondition holds for a general op to execute
static bool isReadyToExecute(ArrayRef<mlir::Value> ins,
                             ArrayRef<mlir::Value> outs,
                             llvm::DenseMap<mlir::Value, llvm::Any> &valueMap) {
  for (auto in : ins)
    if (valueMap.count(in) == 0)
      return false;

  for (auto out : outs)
    if (valueMap.count(out) > 0)
      return false;

  return true;
}

// Fetch values from the value map and consume them
static std::vector<llvm::Any>
fetchValues(ArrayRef<mlir::Value> values,
            llvm::DenseMap<mlir::Value, llvm::Any> &valueMap) {
  std::vector<llvm::Any> ins;
  for (auto &value : values) {
    assert(valueMap[value].has_value());
    ins.push_back(valueMap[value]);
    valueMap.erase(value);
  }
  return ins;
}

// Store values to the value map
static void storeValues(std::vector<llvm::Any> &values,
                        ArrayRef<mlir::Value> outs,
                        llvm::DenseMap<mlir::Value, llvm::Any> &valueMap) {
  assert(values.size() == outs.size());
  for (unsigned long i = 0; i < outs.size(); ++i)
    valueMap[outs[i]] = values[i];
}

// Update the time map after the execution
static void updateTime(ArrayRef<mlir::Value> ins, ArrayRef<mlir::Value> outs,
                       llvm::DenseMap<mlir::Value, double> &timeMap,
                       double latency) {
  double time = 0;
  for (auto &in : ins)
    time = std::max(time, timeMap[in]);
  time += latency;
  for (auto &out : outs)
    timeMap[out] = time;
}

static bool tryToExecute(Operation *op,
                         llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                         llvm::DenseMap<mlir::Value, double> &timeMap,
                         std::vector<mlir::Value> &scheduleList,
                         double latency) {
  auto ins = toVector(op->getOperands());
  auto outs = toVector(op->getResults());

  if (isReadyToExecute(ins, outs, valueMap)) {
    auto in = fetchValues(ins, valueMap);
    std::vector<llvm::Any> out(outs.size());
    auto generalOp = dyn_cast<GeneralOpInterface>(op);
    if (!generalOp)
      op->emitOpError("Undefined execution for the current op");
    generalOp.execute(in, out);
    storeValues(out, outs, valueMap);
    updateTime(ins, outs, timeMap, latency);
    scheduleList = outs;
    return true;
  }
  return false;
}

namespace circt {
namespace handshake {

bool ForkOp::tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                        llvm::DenseMap<unsigned, unsigned> & /*memoryMap*/,
                        llvm::DenseMap<mlir::Value, double> &timeMap,
                        std::vector<std::vector<llvm::Any>> & /*store*/,
                        std::vector<mlir::Value> &scheduleList) {
  return tryToExecute(getOperation(), valueMap, timeMap, scheduleList, 1);
}

bool MergeOp::tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                         llvm::DenseMap<unsigned, unsigned> & /*memoryMap*/,
                         llvm::DenseMap<mlir::Value, double> &timeMap,
                         std::vector<std::vector<llvm::Any>> & /*store*/,
                         std::vector<mlir::Value> &scheduleList) {
  bool found = false;
  for (mlir::Value in : getOperands()) {
    if (valueMap.count(in) == 1) {
      if (found)
        emitOpError("More than one valid input to Merge!");
      auto t = valueMap[in];
      valueMap[getResult()] = t;
      timeMap[getResult()] = timeMap[in];
      // Consume the inputs.
      valueMap.erase(in);
      found = true;
    }
  }
  if (!found)
    emitOpError("No valid input to Merge!");
  scheduleList.push_back(getResult());
  return true;
}
bool MuxOp::tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                       llvm::DenseMap<unsigned, unsigned> & /*memoryMap*/,
                       llvm::DenseMap<mlir::Value, double> &timeMap,
                       std::vector<std::vector<llvm::Any>> & /*store*/,
                       std::vector<mlir::Value> &scheduleList) {
  mlir::Value control = getSelectOperand();
  if (valueMap.count(control) == 0)
    return false;
  auto controlValue = valueMap[control];
  auto controlTime = timeMap[control];
  auto opIdx = llvm::any_cast<APInt>(controlValue).getZExtValue();
  assert(opIdx < getDataOperands().size() &&
         "Trying to select a non-existing mux operand");

  mlir::Value in = getDataOperands()[opIdx];
  if (valueMap.count(in) == 0)
    return false;
  auto inValue = valueMap[in];
  auto inTime = timeMap[in];
  double time = std::max(controlTime, inTime);
  valueMap[getResult()] = inValue;
  timeMap[getResult()] = time;

  // Consume the inputs.
  valueMap.erase(control);
  valueMap.erase(in);
  scheduleList.push_back(getResult());
  return true;
}
bool ControlMergeOp::tryExecute(
    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> & /*memoryMap*/,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> & /*store*/,
    std::vector<mlir::Value> &scheduleList) {
  bool found = false;
  for (auto in : llvm::enumerate(getOperands())) {
    if (valueMap.count(in.value()) == 1) {
      if (found)
        emitOpError("More than one valid input to CMerge!");
      valueMap[getResult()] = valueMap[in.value()];
      timeMap[getResult()] = timeMap[in.value()];

      valueMap[getIndex()] = APInt(INDEX_WIDTH, in.index());
      timeMap[getIndex()] = timeMap[in.value()];

      // Consume the inputs.
      valueMap.erase(in.value());

      found = true;
    }
  }
  if (!found)
    emitOpError("No valid input to CMerge!");
  scheduleList = toVector(getResults());
  return true;
}

void BranchOp::execute(std::vector<llvm::Any> &ins,
                       std::vector<llvm::Any> &outs) {
  outs[0] = ins[0];
}

bool BranchOp::tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                          llvm::DenseMap<unsigned, unsigned> & /*memoryMap*/,
                          llvm::DenseMap<mlir::Value, double> &timeMap,
                          std::vector<std::vector<llvm::Any>> & /*store*/,
                          std::vector<mlir::Value> &scheduleList) {
  return tryToExecute(getOperation(), valueMap, timeMap, scheduleList, 0);
}

bool ConditionalBranchOp::tryExecute(
    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> & /*memoryMap*/,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> & /*store*/,
    std::vector<mlir::Value> &scheduleList) {
  mlir::Value control = getConditionOperand();
  if (valueMap.count(control) == 0)
    return false;
  auto controlValue = valueMap[control];
  auto controlTime = timeMap[control];
  mlir::Value in = getDataOperand();
  if (valueMap.count(in) == 0)
    return false;
  auto inValue = valueMap[in];
  auto inTime = timeMap[in];
  mlir::Value out = llvm::any_cast<APInt>(controlValue) != 0 ? getTrueResult()
                                                             : getFalseResult();
  double time = std::max(controlTime, inTime);
  valueMap[out] = inValue;
  timeMap[out] = time;
  scheduleList.push_back(out);

  // Consume the inputs.
  valueMap.erase(control);
  valueMap.erase(in);
  return true;
}

bool SinkOp::tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                        llvm::DenseMap<unsigned, unsigned> & /*memoryMap*/,
                        llvm::DenseMap<mlir::Value, double> & /*timeMap*/,
                        std::vector<std::vector<llvm::Any>> & /*store*/,
                        std::vector<mlir::Value> & /*scheduleList*/) {
  valueMap.erase(getOperand());
  return true;
}

void BufferOp::execute(std::vector<llvm::Any> &ins,
                       std::vector<llvm::Any> &outs) {
  outs[0] = ins[0];
}

bool BufferOp::tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                          llvm::DenseMap<unsigned, unsigned> & /*memoryMap*/,
                          llvm::DenseMap<mlir::Value, double> &timeMap,
                          std::vector<std::vector<llvm::Any>> & /*store*/,
                          std::vector<mlir::Value> &scheduleList) {
  return tryToExecute(getOperation(), valueMap, timeMap, scheduleList,
                      getNumSlots());
}

void ConstantOp::execute(std::vector<llvm::Any> & /*ins*/,
                         std::vector<llvm::Any> &outs) {
  auto attr = (*this)->getAttrOfType<mlir::IntegerAttr>("value");
  outs[0] = attr.getValue();
}

bool ConstantOp::tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                            llvm::DenseMap<unsigned, unsigned> & /*memoryMap*/,
                            llvm::DenseMap<mlir::Value, double> &timeMap,
                            std::vector<std::vector<llvm::Any>> & /*store*/,
                            std::vector<mlir::Value> &scheduleList) {
  return tryToExecute(getOperation(), valueMap, timeMap, scheduleList, 0);
}

template <typename TMemOp>
static bool
executeMemoryOperation(TMemOp op, unsigned buffer, int opIndex,
                       llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                       llvm::DenseMap<unsigned, unsigned> &memoryMap,
                       llvm::DenseMap<mlir::Value, double> &timeMap,
                       std::vector<std::vector<llvm::Any>> &store,
                       std::vector<mlir::Value> &scheduleList) {
  bool notReady = false;
  for (unsigned i = 0; i < op.getStCount(); i++) {
    mlir::Value data = op->getOperand(opIndex++);
    mlir::Value address = op->getOperand(opIndex++);
    mlir::Value nonceOut = op->getResult(op.getLdCount() + i);
    if ((!valueMap.count(data) || !valueMap.count(address))) {
      notReady = true;
      continue;
    }
    auto addressValue = valueMap[address];
    auto addressTime = timeMap[address];
    auto dataValue = valueMap[data];
    auto dataTime = timeMap[data];

    assert(buffer < store.size());
    auto &ref = store[buffer];
    unsigned offset = llvm::any_cast<APInt>(addressValue).getZExtValue();
    assert(offset < ref.size());
    ref[offset] = dataValue;

    // Implicit none argument
    APInt apnonearg(1, 0);
    valueMap[nonceOut] = apnonearg;
    double time = std::max(addressTime, dataTime);
    timeMap[nonceOut] = time;
    scheduleList.push_back(nonceOut);
    // Consume the inputs.
    valueMap.erase(data);
    valueMap.erase(address);
  }

  for (unsigned i = 0; i < op.getLdCount(); i++) {
    mlir::Value address = op->getOperand(opIndex++);
    mlir::Value dataOut = op->getResult(i);
    mlir::Value nonceOut = op->getResult(op.getLdCount() + op.getStCount() + i);
    if (!valueMap.count(address)) {
      notReady = true;
      continue;
    }
    auto addressValue = valueMap[address];
    auto addressTime = timeMap[address];
    assert(buffer < store.size());
    auto &ref = store[buffer];
    unsigned offset = llvm::any_cast<APInt>(addressValue).getZExtValue();
    assert(offset < ref.size());

    valueMap[dataOut] = ref[offset];
    timeMap[dataOut] = addressTime;
    // Implicit none argument
    APInt apnonearg(1, 0);
    valueMap[nonceOut] = apnonearg;
    timeMap[nonceOut] = addressTime;
    scheduleList.push_back(dataOut);
    scheduleList.push_back(nonceOut);
    // Consume the inputs.
    valueMap.erase(address);
  }
  return (notReady) ? false : true;
}

bool MemoryOp::tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                          llvm::DenseMap<unsigned, unsigned> &memoryMap,
                          llvm::DenseMap<mlir::Value, double> &timeMap,
                          std::vector<std::vector<llvm::Any>> &store,
                          std::vector<mlir::Value> &scheduleList) {
  unsigned buffer = memoryMap[getId()];
  return executeMemoryOperation(*this, buffer, 0, valueMap, memoryMap, timeMap,
                                store, scheduleList);
}

bool LoadOp::tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                        llvm::DenseMap<unsigned, unsigned> & /*memoryMap*/,
                        llvm::DenseMap<mlir::Value, double> &timeMap,
                        std::vector<std::vector<llvm::Any>> & /*store*/,
                        std::vector<mlir::Value> &scheduleList) {
  mlir::Value address = getOperand(0);
  mlir::Value data = getOperand(1);
  mlir::Value nonce = getOperand(2);
  mlir::Value addressOut = getResult(1);
  mlir::Value dataOut = getResult(0);
  if ((valueMap.count(address) && !valueMap.count(nonce)) ||
      (!valueMap.count(address) && valueMap.count(nonce)) ||
      (!valueMap.count(address) && !valueMap.count(nonce) &&
       !valueMap.count(data)))
    return false;
  if (valueMap.count(address) && valueMap.count(nonce)) {
    auto addressValue = valueMap[address];
    auto addressTime = timeMap[address];
    auto nonceValue = valueMap[nonce];
    auto nonceTime = timeMap[nonce];
    valueMap[addressOut] = addressValue;
    double time = std::max(addressTime, nonceTime);
    timeMap[addressOut] = time;
    scheduleList.push_back(addressOut);
    // Consume the inputs.
    valueMap.erase(address);
    valueMap.erase(nonce);
  } else if (valueMap.count(data)) {
    auto dataValue = valueMap[data];
    auto dataTime = timeMap[data];
    valueMap[dataOut] = dataValue;
    timeMap[dataOut] = dataTime;
    scheduleList.push_back(dataOut);
    // Consume the inputs.
    valueMap.erase(data);
  } else {
    llvm_unreachable("why?");
  }
  return true;
}

bool ExternalMemoryOp::tryExecute(
    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    llvm::DenseMap<mlir::Value, double> &timeMap,
    std::vector<std::vector<llvm::Any>> &store,
    std::vector<mlir::Value> &scheduleList) {
  unsigned buffer = llvm::any_cast<unsigned>(valueMap[getMemref()]);
  return executeMemoryOperation(*this, buffer, 1, valueMap, memoryMap, timeMap,
                                store, scheduleList);
}

bool StoreOp::tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                         llvm::DenseMap<unsigned, unsigned> & /*memoryMap*/,
                         llvm::DenseMap<mlir::Value, double> &timeMap,
                         std::vector<std::vector<llvm::Any>> & /*store*/,
                         std::vector<mlir::Value> &scheduleList) {
  return tryToExecute(getOperation(), valueMap, timeMap, scheduleList, 1);
}

void JoinOp::execute(std::vector<llvm::Any> &ins,
                     std::vector<llvm::Any> &outs) {
  outs[0] = ins[0];
}

bool JoinOp::tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                        llvm::DenseMap<unsigned, unsigned> & /*memoryMap*/,
                        llvm::DenseMap<mlir::Value, double> &timeMap,
                        std::vector<std::vector<llvm::Any>> & /*store*/,
                        std::vector<mlir::Value> &scheduleList) {
  return tryToExecute(getOperation(), valueMap, timeMap, scheduleList, 1);
}

void SyncOp::execute(std::vector<llvm::Any> &ins,
                     std::vector<llvm::Any> &outs) {
  outs = ins;
}

bool SyncOp::tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                        llvm::DenseMap<unsigned, unsigned> & /*memoryMap*/,
                        llvm::DenseMap<mlir::Value, double> &timeMap,
                        std::vector<std::vector<llvm::Any>> & /*store*/,
                        std::vector<mlir::Value> &scheduleList) {
  return tryToExecute(getOperation(), valueMap, timeMap, scheduleList, 1);
}

void StoreOp::execute(std::vector<llvm::Any> &ins,
                      std::vector<llvm::Any> &outs) {
  // Forward the address and data to the memory op.
  outs[0] = ins[1];
  outs[1] = ins[0];
}

void ForkOp::execute(std::vector<llvm::Any> &ins,
                     std::vector<llvm::Any> &outs) {
  for (auto &out : outs)
    out = ins[0];
}

bool UnpackOp::tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                          llvm::DenseMap<unsigned, unsigned> & /*memoryMap*/,
                          llvm::DenseMap<mlir::Value, double> &timeMap,
                          std::vector<std::vector<llvm::Any>> & /*store*/,
                          std::vector<mlir::Value> &scheduleList) {
  return tryToExecute(getOperation(), valueMap, timeMap, scheduleList, 1);
}

void UnpackOp::execute(std::vector<llvm::Any> &ins,
                       std::vector<llvm::Any> &outs) {
  auto ins0Vec = llvm::any_cast<std::vector<llvm::Any>>(ins[0]);
  assert(ins0Vec.size() == getNumResults() &&
         "expected that the number of tuple elements matches the number of "
         "outputs");
  for (auto [in, out] : llvm::zip(ins0Vec, outs))
    out = in;
}

bool PackOp::tryExecute(llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                        llvm::DenseMap<unsigned, unsigned> & /*memoryMap*/,
                        llvm::DenseMap<mlir::Value, double> &timeMap,
                        std::vector<std::vector<llvm::Any>> & /*store*/,
                        std::vector<mlir::Value> &scheduleList) {
  return tryToExecute(getOperation(), valueMap, timeMap, scheduleList, 1);
}

void PackOp::execute(std::vector<llvm::Any> &ins,
                     std::vector<llvm::Any> &outs) {
  assert(ins.size() == getNumOperands() &&
         "expected that the number inputs match the number of tuple elements");
  outs[0] = ins;
}

} // namespace handshake
} // namespace circt

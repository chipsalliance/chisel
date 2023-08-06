//===- PrintStateInfo.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/ToolOutputFile.h"

#define DEBUG_TYPE "arc-print-state-info"

using namespace mlir;
using namespace circt;
using namespace arc;
using namespace hw;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct StateInfo {
  enum Type { Input, Output, Register, Memory, Wire } type;
  StringAttr name;
  unsigned offset;
  unsigned numBits;
  unsigned memoryStride = 0; // byte separation between memory words
  unsigned memoryDepth = 0;  // number of words in a memory
};

struct ModelInfo {
  size_t numStateBytes;
  std::vector<StateInfo> states;
};

struct PrintStateInfoPass : public PrintStateInfoBase<PrintStateInfoPass> {
  void runOnOperation() override;
  LogicalResult runOnOperation(llvm::raw_ostream &outputStream);
  LogicalResult collectStates(Value storage, unsigned offset,
                              std::vector<StateInfo> &stateInfos);

  using PrintStateInfoBase::stateFile;
};
} // namespace

void PrintStateInfoPass::runOnOperation() {
  // Print to the output file if one was given, or stdout otherwise.
  if (stateFile.empty()) {
    auto result = runOnOperation(llvm::outs());
    llvm::outs() << "\n";
    if (failed(result))
      return signalPassFailure();
  } else {
    std::error_code ec;
    llvm::ToolOutputFile outputFile(stateFile, ec,
                                    llvm::sys::fs::OpenFlags::OF_None);
    if (ec) {
      mlir::emitError(getOperation().getLoc(), "unable to open state file: ")
          << ec.message();
      return signalPassFailure();
    }
    if (failed(runOnOperation(outputFile.os())))
      return signalPassFailure();
    outputFile.keep();
  }
}

LogicalResult
PrintStateInfoPass::runOnOperation(llvm::raw_ostream &outputStream) {
  llvm::json::OStream json(outputStream, 2);
  bool anyFailed = false;
  json.array([&] {
    std::vector<StateInfo> states;
    for (auto modelOp : getOperation().getOps<ModelOp>()) {
      auto storageArg = modelOp.getBody().getArgument(0);
      auto storageType = storageArg.getType().cast<StorageType>();
      states.clear();
      if (failed(collectStates(storageArg, 0, states))) {
        anyFailed = true;
        return;
      }
      llvm::sort(states, [](auto &a, auto &b) { return a.offset < b.offset; });

      json.object([&] {
        json.attribute("name", modelOp.getName());
        json.attribute("numStateBytes", storageType.getSize());
        json.attributeArray("states", [&] {
          for (const auto &state : states) {
            json.object([&] {
              json.attribute("name", state.name.getValue());
              json.attribute("offset", state.offset);
              json.attribute("numBits", state.numBits);
              auto typeStr = [](StateInfo::Type type) {
                switch (type) {
                case StateInfo::Input:
                  return "input";
                case StateInfo::Output:
                  return "output";
                case StateInfo::Register:
                  return "register";
                case StateInfo::Memory:
                  return "memory";
                case StateInfo::Wire:
                  return "wire";
                }
                return "";
              };
              json.attribute("type", typeStr(state.type));
              if (state.type == StateInfo::Memory) {
                json.attribute("stride", state.memoryStride);
                json.attribute("depth", state.memoryDepth);
              }
            });
          }
        });
      });
    }
  });
  return failure(anyFailed);
}

LogicalResult
PrintStateInfoPass::collectStates(Value storage, unsigned offset,
                                  std::vector<StateInfo> &stateInfos) {
  for (auto *op : storage.getUsers()) {
    if (auto substorage = dyn_cast<AllocStorageOp>(op)) {
      if (!substorage.getOffset().has_value()) {
        substorage.emitOpError(
            "without allocated offset; run state allocation first");
        return failure();
      }
      if (failed(collectStates(substorage.getOutput(),
                               *substorage.getOffset() + offset, stateInfos)))
        return failure();
      continue;
    }
    if (!isa<AllocStateOp, RootInputOp, RootOutputOp, AllocMemoryOp>(op))
      continue;
    auto opName = op->getAttrOfType<StringAttr>("name");
    if (!opName || opName.getValue().empty())
      continue;
    auto opOffset = op->getAttrOfType<IntegerAttr>("offset");
    if (!opOffset) {
      op->emitOpError("without allocated offset; run state allocation first");
      return failure();
    }
    if (isa<AllocStateOp, RootInputOp, RootOutputOp>(op)) {
      auto result = op->getResult(0);
      auto &stateInfo = stateInfos.emplace_back();
      stateInfo.type = StateInfo::Register;
      if (isa<RootInputOp>(op))
        stateInfo.type = StateInfo::Input;
      else if (isa<RootOutputOp>(op))
        stateInfo.type = StateInfo::Output;
      else if (auto alloc = dyn_cast<AllocStateOp>(op)) {
        if (alloc.getTap())
          stateInfo.type = StateInfo::Wire;
      }
      stateInfo.name = opName;
      stateInfo.offset = opOffset.getValue().getZExtValue() + offset;
      stateInfo.numBits =
          result.getType().cast<StateType>().getType().getWidth();
      continue;
    }
    if (auto memOp = dyn_cast<AllocMemoryOp>(op)) {
      auto stride = op->getAttrOfType<IntegerAttr>("stride");
      if (!stride) {
        op->emitOpError("without allocated stride; run state allocation first");
        return failure();
      }
      auto memType = memOp.getType();
      auto intType = memType.getWordType();
      auto &stateInfo = stateInfos.emplace_back();
      stateInfo.type = StateInfo::Memory;
      stateInfo.name = opName;
      stateInfo.offset = opOffset.getValue().getZExtValue() + offset;
      stateInfo.numBits = intType.getWidth();
      stateInfo.memoryStride = stride.getValue().getZExtValue();
      stateInfo.memoryDepth = memType.getNumWords();
      continue;
    }
  }
  return success();
}

std::unique_ptr<Pass> arc::createPrintStateInfoPass(StringRef stateFile) {
  auto pass = std::make_unique<PrintStateInfoPass>();
  if (!stateFile.empty())
    pass->stateFile.assign(stateFile);
  return pass;
}

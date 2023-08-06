//===- Engine.h - LLHD simulaton engine -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the main Engine class of the LLHD simulator.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LLHD_SIMULATOR_ENGINE_H
#define CIRCT_DIALECT_LLHD_SIMULATOR_ENGINE_H

#include "State.h"
#include "Trace.h"

#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
class ExecutionEngine;
} // namespace mlir

namespace llvm {
class Error;
class Module;
} // namespace llvm

namespace circt {
namespace llhd {
namespace sim {

class Engine {
public:
  /// Initialize an LLHD simulation engine. This initializes the state, as well
  /// as the mlir::ExecutionEngine with the given module.
  Engine(
      llvm::raw_ostream &out, ModuleOp module,
      llvm::function_ref<mlir::LogicalResult(mlir::ModuleOp)> mlirTransformer,
      llvm::function_ref<llvm::Error(llvm::Module *)> llvmTransformer,
      std::string root, TraceMode tm, ArrayRef<StringRef> sharedLibPaths);

  /// Default destructor
  ~Engine();

  /// Run simulation up to n steps or maxTime picoseconds of simulation time.
  /// n=0 and T=0 make the simulation run indefinitely.
  int simulate(int n, uint64_t maxTime);

  /// Build the instance layout of the design.
  void buildLayout(ModuleOp module);

  /// Get the MLIR module.
  const ModuleOp getModule() const { return module; }

  /// Get the simulation state.
  const State *getState() const { return state.get(); }

  /// Dump the instance layout stored in the State.
  void dumpStateLayout();

  /// Dump the instances each signal triggers.
  void dumpStateSignalTriggers();

private:
  void walkEntity(EntityOp entity, Instance &child);

  llvm::raw_ostream &out;
  std::string root;
  std::unique_ptr<State> state;
  std::unique_ptr<mlir::ExecutionEngine> engine;
  ModuleOp module;
  TraceMode traceMode;
};

} // namespace sim
} // namespace llhd
} // namespace circt

#endif // CIRCT_DIALECT_LLHD_SIMULATOR_ENGINE_H

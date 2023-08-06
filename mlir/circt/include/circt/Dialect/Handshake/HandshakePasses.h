//===- HandshakePasses.h - Handshake pass entry points ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HANDSHAKE_HANDSHAKEPASSES_H
#define CIRCT_DIALECT_HANDSHAKE_HANDSHAKEPASSES_H

#include "circt/Support/LLVM.h"
#include <map>
#include <memory>
#include <optional>
#include <set>

namespace circt {
namespace handshake {
class FuncOp;

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHandshakeDotPrintPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHandshakeOpCountPass();
std::unique_ptr<mlir::Pass> createHandshakeMaterializeForksSinksPass();
std::unique_ptr<mlir::Pass> createHandshakeDematerializeForksSinksPass();
std::unique_ptr<mlir::Pass> createHandshakeRemoveBuffersPass();
std::unique_ptr<mlir::Pass> createHandshakeAddIDsPass();
std::unique_ptr<mlir::Pass>
createHandshakeLowerExtmemToHWPass(std::optional<bool> createESIWrapper = {});
std::unique_ptr<mlir::Pass> createHandshakeLegalizeMemrefsPass();
std::unique_ptr<mlir::OperationPass<handshake::FuncOp>>
createHandshakeInsertBuffersPass(const std::string &strategy = "all",
                                 unsigned bufferSize = 2);
std::unique_ptr<mlir::Pass> createHandshakeLockFunctionsPass();

/// Iterates over the handshake::FuncOp's in the program to build an instance
/// graph. In doing so, we detect whether there are any cycles in this graph, as
/// well as infer a top function for the design by performing a topological sort
/// of the instance graph. The result of this sort is placed in sortedFuncs.
using InstanceGraph = std::map<std::string, std::set<std::string>>;
LogicalResult resolveInstanceGraph(ModuleOp moduleOp,
                                   InstanceGraph &instanceGraph,
                                   std::string &topLevel,
                                   SmallVectorImpl<std::string> &sortedFuncs);

// Checks all block arguments and values within op to ensure that all
// values have exactly one use.
LogicalResult verifyAllValuesHasOneUse(handshake::FuncOp op);

// Adds sink operations to any unused value in r.
LogicalResult addSinkOps(Region &r, OpBuilder &rewriter);

// Adds fork operations to any value with multiple uses in r.
LogicalResult addForkOps(Region &r, OpBuilder &rewriter);
void insertFork(Value result, bool isLazy, OpBuilder &rewriter);

// Adds a locking mechanism around the region.
LogicalResult lockRegion(Region &r, OpBuilder &rewriter);

// Applies the spcified buffering strategy on the region r.
LogicalResult bufferRegion(Region &r, OpBuilder &rewriter, StringRef strategy,
                           unsigned bufferSize);

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Handshake/HandshakePasses.h.inc"

} // namespace handshake
} // namespace circt

#endif // CIRCT_DIALECT_HANDSHAKE_HANDSHAKEPASSES_H

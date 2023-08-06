//===-- Passes.td - Transforms pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes within the Transforms/ directory.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TRANSFORMS_PASSES
#define CIRCT_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def FlattenMemRef : Pass<"flatten-memref", "::mlir::ModuleOp"> {
  let summary = "Flatten memrefs";
  let description = [{
    Flattens multidimensional memories and accesses to them into
    single-dimensional memories.}];
  let constructor = "circt::createFlattenMemRefPass()";
  let dependentDialects = ["mlir::memref::MemRefDialect"];
}

def FlattenMemRefCalls : Pass<"flatten-memref-calls", "::mlir::ModuleOp"> {
  let summary = "Flatten memref calls";
  let description = [{
    Flattens calls to functions which have multidimensional memrefs as arguments.
    This is done by casting memref arguments through memref.subview operations.
    Any called functions which had their type signatures changes will be replaced
    by a private function definition with the new signature.
    It is up to users of this pass to define how these rewritten functions are
    to be implemented.}];
  let constructor = "circt::createFlattenMemRefCallsPass()";
  let dependentDialects = ["mlir::memref::MemRefDialect"];
}

def StripDebugInfoWithPred : Pass<"strip-debuginfo-with-pred", "::mlir::ModuleOp"> {
  let summary = "Selectively strip debug info from all operations";

  let description = [{
    This pass extends mlir::StripDebugInfoPass to selectively strip locations with a
    given predicate.
  }];
  let constructor = "circt::createStripDebugInfoWithPredPass({})";
  let options = [
    Option<"dropSuffix", "drop-suffix", "std::string",
           /*default=*/"",
           "Drop file location info with the specified suffix. This option is"
           "intended to be used for testing."> ];
}

def MapArithToCombPass : InterfacePass<"map-arith-to-comb", "circt::hw::HWModuleLike"> {
  let summary = "Map arith ops to combinational logic";
  let description = [{
    A pass which does a simple `arith` to `comb` mapping wherever possible.
    This pass will not convert:  
    * floating point operations
    * operations using `vector`-typed values

    This **does not** intend to be the definitive lowering/HLS pass of `arith`
    operations in CIRCT (hence the name "map" instead of e.g. "lower").
    Rather, it provides a simple way (mostly for testing purposes) to map
    `arith` operations.
  }];
  let constructor = "circt::createMapArithToCombPass()";
  let dependentDialects = ["circt::hw::HWDialect, mlir::arith::ArithDialect, circt::comb::CombDialect"];
}

#endif // CIRCT_TRANSFORMS_PASSES

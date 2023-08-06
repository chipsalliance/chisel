//===-- Passes.td - LLHD pass definition file --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes that work on the LLHD dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LLHD_TRANSFORMS_PASSES
#define CIRCT_DIALECT_LLHD_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def ProcessLowering : Pass<"llhd-process-lowering", "ModuleOp"> {
  let summary = "Lowers LLHD Processes to Entities.";
  let description = [{
    TODO
  }];

  let constructor = "circt::llhd::createProcessLoweringPass()";
}

def MemoryToBlockArgument : Pass<"llhd-memory-to-block-argument",
                                 "llhd::ProcOp"> {
  let summary = "Promote memory to block arguments.";
  let description = [{
    Promotes memory locations allocated with `llhd.var` to block arguments. This
    enables other optimizations and is required to be able to lower behavioral
    LLHD to structural LLHD. This is because there are no memory model and
    control flow in structural LLHD. After executing this pass, the
    "-llhd-block-argument-to-mux" pass can be used to convert the block
    arguments to multiplexers to enable more control-flow elimination.

    Example:

    ```mlir
    llhd.proc @check_simple(%condsig : !llhd.sig<i1>) -> () {
      %c5 = llhd.const 5 : i32
      %cond = llhd.prb %condsig : !llhd.sig<i1>
      %ptr = llhd.var %c5 : i32
      cond_br %cond, ^bb1, ^bb2
    ^bb1:
      %c6 = llhd.const 6 : i32
      llhd.store %ptr, %c6 : !llhd.ptr<i32>
      br ^bb2
    ^bb2:
      %ld = llhd.load %ptr : !llhd.ptr<i32>
      %res = llhd.not %ld : i32
      llhd.halt
    }
    ```

    is transformed to

    ```mlir
    llhd.proc @check_simple(%condsig : !llhd.sig<i1>) -> () {
      %c5 = llhd.const 5 : i32
      %cond = llhd.prb %condsig : !llhd.sig<i1>
      cond_br %cond, ^bb1, ^bb2(%c5 : i32)
    ^bb1:
      %c6 = llhd.const 6 : i32
      br ^bb2(%c6 : i32)
    ^bb2(%arg : i32):
      %res = llhd.not %arg : i32
      llhd.halt
    }
    ```

  }];

  let constructor = "circt::llhd::createMemoryToBlockArgumentPass()";
}

def FunctionElimination : Pass<"llhd-function-elimination", "ModuleOp"> {
  let summary = "Deletes all functions.";
  let description = [{
    Deletes all functions in the module. In case there is still a function
    call in an entity or process, it fails.
    This pass is intended as a post-inlining pass to check if all functions
    could be successfully inlined and remove the inlined functions. This
    is necessary because Structural LLHD does not allow functions. Fails in
    the case that there is still a function call left in a `llhd.proc` or
    `llhd.entity`.
  }];

  let constructor = "circt::llhd::createFunctionEliminationPass()";
}

def EarlyCodeMotion : Pass<"llhd-early-code-motion", "llhd::ProcOp"> {
  let summary = "Move side-effect-free instructions and llhd.prb up in the CFG";
  let description = [{
    Moves side-effect-free instructions as far up in the CFG as possible. That
    means to the earliest block where all operands are defined. Special care has
    to be given to the `llhd.prb` instruction (which is the only side-effect
    instruction moved by this pass) as it must stay in the same temporal region,
    because otherwise it might sample an older or newer state of the signal.
    This pass is designed as a preparatory pass for the Temporal Code Motion
    pass to be able to move the `llhd.drv` operations in a single TR exiting
    block without having to move operations defining the operands used by the
    `llhd.drv`. It also enables total control flow elimination as the `llhd.prb`
    instructions would not be moved by other canonicalization passes.
  }];

  let constructor = "circt::llhd::createEarlyCodeMotionPass()";
}

#endif // CIRCT_DIALECT_LLHD_TRANSFORMS_PASSES

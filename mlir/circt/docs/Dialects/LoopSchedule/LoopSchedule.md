# LoopSchedule Dialect Rationale

This document describes various design points of the `loopschedule` dialect, why it is
the way it is, and current status. This follows in the spirit of other [MLIR
Rationale docs](https://mlir.llvm.org/docs/Rationale/).

## Introduction

The `loopschedule` dialect provides a collection of ops to represent software-like loops
after scheduling. There are currently two main kinds of loops that can be represented:
pipelined and sequential. Pipelined loops allow multiple iterations of the loop to be
in-flight at a time and have an associated initiation interval (`II`) to specify the number
of cycles between the start of successive loop iterations. In contrast, sequential loops
are guaranteed to only have one iteration in-flight at any given time.

A primary goal of the `loopschedule` dialect, as opposed to many other High-Level Synthesis
(HLS) representations, is to maintain the structure of loops after scheduling. As such, the
`loopschedule` ops are inspired by the `scf` and `affine` dialect ops.

## Pipelined Loops

Pipelined loops are represented with the `loopschedule.pipeline` op. A `pipeline`
loop resembles a `while` loop in the `scf` dialect, but the body must contain only 
`loopschedule.pipeline.stage` and `loopschedule.terminator` ops. To have a better 
understanding of how  `loopschedule.pipeline` works, we will look at the following 
example:

```
func.func @test1(%arg0: memref<10xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = loopschedule.pipeline II = 1 iter_args(%arg1 = %c0, %arg2 = %c0_i32) : (index, i32) -> i32 {
    %1 = arith.cmpi ult, %arg1, %c10 : index
    loopschedule.register %1 : i1
  } do {
    %1:2 = loopschedule.pipeline.stage start = 0 {
      %3 = arith.addi %arg1, %c1 : index
      %4 = memref.load %arg0[%arg1] : memref<10xi32>
      loopschedule.register %3, %4 : index, i32
    } : index, i32
    %2 = loopschedule.pipeline.stage start = 1 {
      %3 = arith.addi %1#1, %arg2 : i32
      pipeline.register %3 : i32
    } : i32
    loopschedule.terminator iter_args(%1#0, %2), results(%2) : (index, i32) -> i32
  }
  return %0 : i32
}
```

A `pipeline` op first defines the initial values for the `iter_args`. `iter_args` are values that will
be passed back to the first stage after the last stage of the pipeline. The pipeline also defines a
specific, static `II`. Each pipeline stage in the `do` block represents a series of ops run in parallel.

Values are registered at the end of a stage and passed out as results for future pipeline stages to
use. Each pipeline stage must have a defined start time, which is the number of cycles between the 
start of the pipeline and when the first valid data will be available as input to that stage.

Finally, the terminator is called with the `iter_args` for the next iteration and the result values
that will be returned when the pipeline completes. Even though the terminator is located at the
end of the loop body, its values are passed back to a previous stage whenever needed. We do not
need to wait for an entire iteration to finish before `iter_args` become valid for the next iteration.

Multi-cycle and pipelined ops can also be supported in `pipeline` loops. In the following example,
assume the multiply op is bound to a 3-stage pipelined multiplier:

```
func.func @test1(%arg0: memref<10xi32>, %arg1: memref<10xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c1_i32 = arith.constant 1 : i32
  loopschedule.pipeline II = 1 iter_args(%arg2 = %c0) : (index, i32) -> () {
    %1 = arith.cmpi ult, %arg1, %c10 : index
    loopschedule.register %1 : i1
  } do {
    %1:2 = loopschedule.pipeline.stage start = 0 {
      %3 = arith.addi %arg1, %c1 : index
      %4 = memref.load %arg0[%arg2] : memref<10xi32>
      loopschedule.register %3, %4 : index, i32
    } : index, i32
    %2:2 = loopschedule.pipeline.stage start = 1 {
      %3 = arith.muli %1#1, %c1_i32 : i32
      pipeline.register %3, %1#0 : i32
    } : i32
    loopschedule.pipeline.stage start = 4 {
      memref.store %2#0, %arg0[%2#1] : i32
      pipeline.register
    } : i32
    loopschedule.terminator iter_args(%1#0), results() : (index, i32) -> ()
  }
  return
}
```

Here, the `II` is still 1 because new values can be introduced to the multiplier every cycle. The last
stage is delayed by 3 cycles because of the 3 cycle latency of the multiplier. The `pipeline` op is 
currently tightly coupled to the lowering implementation used, as the latency of operators is not 
represented in the IR, but rather an implicit assumption made when lowering later. The scheduling 
problem is constructed with these implicit operator latencies in mind. This coupling can be addressed 
in the future with a proper operator library to maintain explicit operator latencies in the IR.

## Status

Added pipeline loop representation, more documentation and rationale to come as ops are added.

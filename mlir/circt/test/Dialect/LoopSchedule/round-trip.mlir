// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

func.func @test1(%arg0: memref<?xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c0_i32 = arith.constant 0 : i32
  // CHECK: loopschedule.pipeline
  // CHECK-SAME: II = 1
  // CHECK-SAME: iter_args(%arg1 = %c0, %arg2 = %c0_i32)
  // CHECK-SAME: (index, i32) -> i32
  // CHECK-SAME: {
  %0 = loopschedule.pipeline II = 1 iter_args(%arg1 = %c0, %arg2 = %c0_i32) : (index, i32) -> i32 {
    %1 = arith.cmpi ult, %arg1, %c10 : index
    loopschedule.register %1 : i1
  // CHECK: } do {
  } do {
    // CHECK: loopschedule.pipeline.stage start = 0 {
    %1:2 = loopschedule.pipeline.stage start = 0 {
      %3 = arith.addi %arg1, %c1 : index
      %4 = memref.load %arg0[%arg1] : memref<?xi32>
      // CHECK: loopschedule.register {{.+}} : {{.+}}
      // CHECK-NEXT: } : index, i32
      loopschedule.register %3, %4 : index, i32
    } : index, i32
    %2 = loopschedule.pipeline.stage start = 1 {
      %3 = arith.addi %1#1, %arg2 : i32
      loopschedule.register %3 : i32
    } : i32
    // CHECK: loopschedule.terminator iter_args({{.+}}), results({{.+}}) : {{.+}}
    loopschedule.terminator iter_args(%1#0, %2), results(%2) : (index, i32) -> i32
  }
  return %0 : i32
}

func.func @test2(%arg0: memref<?xi32>, %arg1: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c10 = arith.constant 10 : index
  // CHECK: loopschedule.pipeline
  // CHECK-SAME: II = 1
  // CHECK-SAME: iter_args(%arg2 = %c0)
  // CHECK-SAME: (index) -> ()
  loopschedule.pipeline II = 1 iter_args(%arg2 = %c0) : (index) -> () {
    %0 = arith.cmpi ult, %arg2, %c10 : index
    loopschedule.register %0 : i1
  } do {
    %0:4 = loopschedule.pipeline.stage start = 0 {
      %4 = arith.addi %arg2, %c1 : index
      %5 = memref.load %arg0[%arg2] : memref<?xi32>
      %6 = arith.cmpi uge, %arg2, %c3 : index
      loopschedule.register %arg2, %4, %5, %6 : index, index, i32, i1
    } : index, index, i32, i1
    // CHECK: loopschedule.pipeline.stage start = 1 when %0#3
    %1:3 = loopschedule.pipeline.stage start = 1 when %0#3  {
      %4 = arith.subi %0#0, %c3 : index
      loopschedule.register %0#2, %0#3, %4 : i32, i1, index
    } : i32, i1, index
    %2:4 = loopschedule.pipeline.stage start = 2 when %1#1  {
      %4 = memref.load %arg0[%1#2] : memref<?xi32>
      loopschedule.register %1#0, %1#1, %1#2, %4 : i32, i1, index, i32
    } : i32, i1, index, i32
    %3:3 = loopschedule.pipeline.stage start = 3 when %2#1  {
      %4 = arith.addi %2#0, %2#3 : i32
      loopschedule.register %2#1, %2#2, %4 : i1, index, i32
    } : i1, index, i32
    loopschedule.pipeline.stage start = 5 when %3#0  {
      memref.store %3#2, %arg1[%3#1] : memref<?xi32>
      loopschedule.register
    }
    loopschedule.terminator iter_args(%0#0), results() : (index) -> ()
  }
  return
}

// CHECK-LABEL: func.func @test3
func.func @test3(%arg0: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = memref.alloca() : memref<1xi32>
  %1 = memref.alloca() : memref<1xi32>
  %2 = memref.alloca() : memref<1xi32>
  // CHECK: loopschedule.pipeline
  // CHECK-SAME: II = 1
  // CHECK-SAME: iter_args(%arg1 = %c0)
  // CHECK-SAME: (index) -> ()
  loopschedule.pipeline II = 1 iter_args(%arg1 = %c0) : (index) -> () {
    %3 = arith.cmpi ult, %arg1, %c10 : index
    loopschedule.register %3 : i1
  } do {
    %3:5 = loopschedule.pipeline.stage start = 0 {
      %5 = arith.addi %arg1, %c1 : index
      %6 = memref.load %2[%c0] : memref<1xi32>
      %7 = memref.load %1[%c0] : memref<1xi32>
      %8 = memref.load %0[%c0] : memref<1xi32>
      %9 = memref.load %arg0[%arg1] : memref<?xi32>
      loopschedule.register %5, %6, %7, %8, %9 : index, i32, i32, i32, i32
    } : index, i32, i32, i32, i32
    %4 = loopschedule.pipeline.stage start = 1 {
      memref.store %3#2, %2[%c0] : memref<1xi32>
      memref.store %3#3, %1[%c0] : memref<1xi32>
      %5 = arith.addi %3#1, %3#4 : i32
      loopschedule.register %5 : i32
    } : i32
    loopschedule.pipeline.stage start = 2 {
      memref.store %4, %0[%c0] : memref<1xi32>
      loopschedule.register
    }
    loopschedule.terminator iter_args(%3#0), results() : (index) -> ()
  }
  return
}

// CHECK-LABEL: func.func @test4
func.func @test4(%arg0: memref<?xi32>, %arg1: memref<?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c1_i32 = arith.constant 1 : i32
  // CHECK: loopschedule.pipeline
  // CHECK-SAME: II = 1
  // CHECK-SAME: iter_args(%arg2 = %c0)
  // CHECK-SAME: (index) -> ()
  loopschedule.pipeline II = 1 iter_args(%arg2 = %c0) : (index) -> () {
    %0 = arith.cmpi ult, %arg2, %c10 : index
    loopschedule.register %0 : i1
  } do {
    %0:2 = loopschedule.pipeline.stage start = 0 {
      %3 = arith.addi %arg2, %c1 : index
      %4 = memref.load %arg1[%arg2] : memref<?xi32>
      %5 = arith.index_cast %4 : i32 to index
      loopschedule.register %3, %5 : index, index
    } : index, index
    %1:2 = loopschedule.pipeline.stage start = 1 {
      %3 = memref.load %arg0[%0#1] : memref<?xi32>
      loopschedule.register %0#1, %3 : index, i32
    } : index, i32
    %2:2 = loopschedule.pipeline.stage start = 2 {
      %3 = arith.addi %1#1, %c1_i32 : i32
      loopschedule.register %1#0, %3 : index, i32
    } : index, i32
    loopschedule.pipeline.stage start = 4 {
      memref.store %2#1, %arg0[%2#0] : memref<?xi32>
      loopschedule.register
    }
    loopschedule.terminator iter_args(%0#0), results() : (index) -> ()
  }
  return
}

// CHECK-LABEL: func.func @test5
func.func @test5(%arg0: memref<?xi32>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c10 = arith.constant 10 : index
  // CHECK: loopschedule.pipeline
  // CHECK-SAME: II = 1
  // CHECK-SAME: iter_args(%arg1 = %c2)
  // CHECK-SAME: (index) -> ()
  loopschedule.pipeline II = 1 iter_args(%arg1 = %c2) : (index) -> () {
    %0 = arith.cmpi ult, %arg1, %c10 : index
    loopschedule.register %0 : i1
  } do {
    %0 = loopschedule.pipeline.stage start = 0 {
      %2 = arith.subi %arg1, %c2 : index
      %3 = memref.load %arg0[%2] : memref<?xi32>
      loopschedule.register %3 : i32
    } : i32
    %1:2 = loopschedule.pipeline.stage start = 1 {
      %2 = arith.subi %arg1, %c1 : index
      %3 = memref.load %arg0[%2] : memref<?xi32>
      %4 = arith.addi %arg1, %c1 : index
      loopschedule.register %3, %4 : i32, index
    } : i32, index
    loopschedule.pipeline.stage start = 2 {
      %2 = arith.addi %0, %1#0 : i32
      memref.store %2, %arg0[%arg1] : memref<?xi32>
      loopschedule.register
    }
    loopschedule.terminator iter_args(%1#1), results() : (index) -> ()
  }
  return
}

func.func @trip_count_attr() {
  %false = arith.constant 0 : i1
  // CHECK: loopschedule.pipeline II = 1 trip_count = 3
  loopschedule.pipeline II = 1 trip_count = 3 iter_args(%arg0 = %false) : (i1) -> () {
    loopschedule.register %arg0 : i1
  } do {
    %0 = loopschedule.pipeline.stage start = 0 {
      loopschedule.register %arg0 : i1
    } : i1
    loopschedule.terminator iter_args(%0), results() : (i1) -> ()
  }
  return
}

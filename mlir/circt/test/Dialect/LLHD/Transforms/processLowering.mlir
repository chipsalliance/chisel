// RUN: circt-opt %s -llhd-process-lowering -split-input-file -verify-diagnostics | FileCheck %s

// no inputs and outputs
// CHECK: llhd.entity @empty () -> () {
llhd.proc @empty() -> () {
  // CHECK-NEXT: }
  llhd.halt
}

// check that input and output signals are transferred correctly
// CHECK-NEXT: llhd.entity @inputAndOutput (%{{.*}} : !llhd.sig<i64>, %{{.*}} : !llhd.sig<i1>) -> (%{{.*}} : !llhd.sig<i1>) {
llhd.proc @inputAndOutput(%arg0 : !llhd.sig<i64>, %arg1 : !llhd.sig<i1>) -> (%arg2 : !llhd.sig<i1>) {
  // CHECK-NEXT: }
  llhd.halt
}

// check wait suspended process
// CHECK-NEXT: llhd.entity @simpleWait () -> () {
llhd.proc @simpleWait() -> () {
  // CHECK-NEXT: }
  cf.br ^bb1
^bb1:
  llhd.wait ^bb1
}

// Check wait with observing probed signals
// CHECK-NEXT: llhd.entity @prbAndWait (%{{.*}} : !llhd.sig<i64>) -> () {
llhd.proc @prbAndWait(%arg0 : !llhd.sig<i64>) -> () {
  // CHECK-NEXT: %{{.*}} = llhd.prb
  // CHECK-NEXT: }
  cf.br ^bb1
^bb1:
  %0 = llhd.prb %arg0 : !llhd.sig<i64>
  llhd.wait (%arg0 : !llhd.sig<i64>), ^bb1
}

// Check wait with observing probed signals
// CHECK-NEXT: llhd.entity @prbAndWaitMoreObserved (%{{.*}} : !llhd.sig<i64>, %{{.*}} : !llhd.sig<i64>) -> () {
llhd.proc @prbAndWaitMoreObserved(%arg0 : !llhd.sig<i64>, %arg1 : !llhd.sig<i64>) -> () {
  // CHECK-NEXT: %{{.*}} = llhd.prb
  // CHECK-NEXT: }
  cf.br ^bb1
^bb1:
  %0 = llhd.prb %arg0 : !llhd.sig<i64>
  llhd.wait (%arg0, %arg1 : !llhd.sig<i64>, !llhd.sig<i64>), ^bb1
}

// CHECK-NEXT: llhd.entity @muxedSignal
llhd.proc @muxedSignal(%arg0 : !llhd.sig<i64>, %arg1 : !llhd.sig<i64>) -> () {
  cf.br ^bb1
^bb1:
  // CHECK-NEXT: %{{.*}} = hw.constant
  // CHECK-NEXT: %{{.*}} = comb.mux
  // CHECK-NEXT: %{{.*}} = llhd.prb
  // CHECK-NEXT: }
  %cond = hw.constant true
  %sig = comb.mux %cond, %arg0, %arg1 : !llhd.sig<i64>
  %0 = llhd.prb %sig : !llhd.sig<i64>
  llhd.wait (%arg0, %arg1 : !llhd.sig<i64>, !llhd.sig<i64>), ^bb1
}

// CHECK-NEXT: llhd.entity @muxedSignal2
llhd.proc @muxedSignal2(%arg0 : !llhd.sig<i64>, %arg1 : !llhd.sig<i64>) -> () {
  cf.br ^bb1
^bb1:
  // CHECK-NEXT: %{{.*}} = hw.constant
  // CHECK-NEXT: %{{.*}} = comb.mux
  // CHECK-NEXT: %{{.*}} = llhd.prb
  // CHECK-NEXT: }
  %cond = hw.constant true
  %sig = comb.mux %cond, %arg0, %arg1 : !llhd.sig<i64>
  %0 = llhd.prb %sig : !llhd.sig<i64>
  llhd.wait (%sig : !llhd.sig<i64>), ^bb1
}

// CHECK-NEXT: llhd.entity @partialSignal
llhd.proc @partialSignal(%arg0 : !llhd.sig<i64>) -> () {
  cf.br ^bb1
^bb1:
  // CHECK-NEXT: %{{.*}} = hw.constant
  // CHECK-NEXT: %{{.*}} = llhd.sig.extract
  // CHECK-NEXT: %{{.*}} = llhd.prb
  // CHECK-NEXT: }
  %c = hw.constant 16 : i6
  %sig = llhd.sig.extract %arg0 from %c : (!llhd.sig<i64>) -> !llhd.sig<i32>
  %0 = llhd.prb %sig : !llhd.sig<i32>
  llhd.wait (%arg0 : !llhd.sig<i64>), ^bb1
}

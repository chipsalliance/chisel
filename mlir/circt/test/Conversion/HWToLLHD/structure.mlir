// RUN: circt-opt --convert-hw-to-llhd --verify-diagnostics %s | FileCheck %s

module {
  // CHECK-LABEL: llhd.entity @FeedThrough
  // CHECK-SAME: (%[[IN:.+]] : !llhd.sig<i1>) ->
  // CHECK-SAME: (%[[OUT:.+]] : !llhd.sig<i1>)
  hw.module @FeedThrough(%in: i1) -> (out: i1) {
    // CHECK: llhd.con %[[OUT]], %[[IN]]
    hw.output %in: i1
  }

  // CHECK-LABEL: llhd.entity @CombOp
  // CHECK-SAME: (%[[IN:.+]] : !llhd.sig<i1>) ->
  // CHECK-SAME: (%[[OUT:.+]] : !llhd.sig<i1>)
  hw.module @CombOp(%in: i1) -> (out: i1) {
    // CHECK: %[[PRB:.+]] = llhd.prb %[[IN]]
    // CHECK: %[[ADD:.+]] = comb.add %[[PRB]], %[[PRB]]
    // CHECK: llhd.drv %[[OUT]], %[[ADD]]
    %0 = comb.add %in, %in : i1
    hw.output %0: i1
  }

  // CHECK-LABEL: llhd.entity @sub
  // CHECK-SAME: ({{.+}} : !llhd.sig<i1>) ->
  // CHECK-SAME: ({{.+}} : !llhd.sig<i1>)
  hw.module @sub(%in: i1) -> (out: i1) {
    hw.output %in : i1
  }

// CHECK-LABEL: llhd.entity @SimpleInstance
  // CHECK-SAME: (%[[IN:.+]] : !llhd.sig<i1>) ->
  // CHECK-SAME: (%[[OUT:.+]] : !llhd.sig<i1>)
  hw.module @SimpleInstance(%in: i1) -> (out: i1) {
    // CHECK: llhd.inst "sub1" @sub(%[[IN]]) -> (%[[OUT]]) : (!llhd.sig<i1>) -> !llhd.sig<i1>
    %0 = hw.instance "sub1" @sub (in: %in: i1) -> (out: i1)
    hw.output %0 : i1
  }

  // CHECK-LABEL: llhd.entity @InstanceWithLocalOps
  // CHECK-SAME: (%[[IN:.+]] : !llhd.sig<i1>) ->
  // CHECK-SAME: (%[[OUT:.+]] : !llhd.sig<i1>)
  hw.module @InstanceWithLocalOps(%in: i1) -> (out: i1) {
    // CHECK: %[[PRB:.+]] = llhd.prb %[[IN]]
    // CHECK: %[[ADD1:.+]] = comb.add %[[PRB]], %[[PRB]]
    // CHECK: llhd.drv %[[ARG_SIG:.+]], %[[ADD1]]
    // CHECK: %[[RES_SIG:.+]] = llhd.sig
    // CHECK: %[[RES:.+]] = llhd.prb %[[RES_SIG]]
    // CHECK: llhd.inst "sub1" @sub(%[[ARG_SIG]]) -> (%[[RES_SIG]]) : (!llhd.sig<i1>) -> !llhd.sig<i1>
    // CHECK: %[[ADD2:.+]] = comb.add %[[RES]], %[[RES]]
    // CHECK: llhd.drv %[[OUT:.+]], %[[ADD2]]
    %0 = comb.add %in, %in : i1
    %1 = hw.instance "sub1" @sub (in: %0: i1) -> (out: i1)
    %2 = comb.add %1, %1 : i1
    hw.output %2 : i1
  }

  // CHECK-LABEL: llhd.entity @InstanceDirectlyDrivingMultipleOutputs
  // CHECK-SAME: (%[[IN:.+]] : !llhd.sig<i1>) -> (
  // CHECK-SAME: %[[OUT1:.+]] : !llhd.sig<i1>,
  // CHECK-SAME: %[[OUT2:.+]] : !llhd.sig<i1>)
  hw.module @InstanceDirectlyDrivingMultipleOutputs(%in: i1) -> (out1: i1, out2: i1) {
    // CHECK: llhd.inst "sub1" @sub(%[[IN]]) -> (%[[OUT2]]) : (!llhd.sig<i1>) -> !llhd.sig<i1>
    // CHECK: llhd.con %[[OUT1]], %[[OUT2]]
    %0 = hw.instance "sub1" @sub (in: %in: i1) -> (out: i1)
    hw.output %0, %0 : i1, i1
  }

  // CHECK-LABEL: llhd.entity @InstanceIndirectlyDrivingMultipleOutputs
  // CHECK-SAME: (%[[IN:.+]] : !llhd.sig<i1>) -> (
  // CHECK-SAME: %[[OUT1:.+]] : !llhd.sig<i1>,
  // CHECK-SAME: %[[OUT2:.+]] : !llhd.sig<i1>)
  hw.module @InstanceIndirectlyDrivingMultipleOutputs(%in: i1) -> (out1: i1, out2: i1) {
    // CHECK: %[[RES_SIG:.+]] = llhd.sig
    // CHECK: %[[RES:.+]] = llhd.prb %[[RES_SIG]]
    // CHECK: llhd.inst "sub1" @sub(%[[IN]]) -> (%[[RES_SIG]]) : (!llhd.sig<i1>) -> !llhd.sig<i1>
    // CHECK: %[[ADD:.+]] = comb.add %[[RES]], %[[RES]]
    // CHECK: llhd.drv %[[OUT1:.+]], %[[ADD]]
    // CHECK: llhd.drv %[[OUT2:.+]], %[[ADD]]
    %0 = hw.instance "sub1" @sub(in: %in: i1) -> (out: i1)
    %1 = comb.add %0, %0 : i1
    hw.output %1, %1 : i1, i1
  }
}

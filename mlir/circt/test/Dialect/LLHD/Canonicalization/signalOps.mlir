// RUN: circt-opt %s -canonicalize='top-down=true region-simplify=true' | FileCheck %s

// CHECK-LABEL: @drv_folding
// CHECK-SAME: %[[SIG:.*]]: !llhd.sig<i32>
// CHECK-SAME: %[[VAL:.*]]: i32
// CHECK-SAME: %[[TIME:.*]]: !llhd.time
// CHECK-SAME: %[[COND:.*]]: i1
func.func @drv_folding(%sig: !llhd.sig<i32>, %val: i32, %time: !llhd.time, %cond: i1) {
  %true = hw.constant 1 : i1
  %false = hw.constant 0 : i1

  // CHECK-NEXT: llhd.drv %[[SIG]], %[[VAL]] after %[[TIME]] :
  llhd.drv %sig, %val after %time : !llhd.sig<i32>
  // CHECK-NEXT: llhd.drv %[[SIG]], %[[VAL]] after %[[TIME]] if %[[COND]] :
  llhd.drv %sig, %val after %time if %cond : !llhd.sig<i32>
  llhd.drv %sig, %val after %time if %false : !llhd.sig<i32>
  // CHECK-NEXT: llhd.drv %[[SIG]], %[[VAL]] after %[[TIME]] :
  llhd.drv %sig, %val after %time if %true : !llhd.sig<i32>

  // CHECK-NEXT: return
  return
}

// CHECK-LABEL: @con_folding
// CHECK-NOT: llhd.con
llhd.entity @con_folding(%sig: !llhd.sig<i32>) {
  llhd.con %sig, %sig : !llhd.sig<i32>
}

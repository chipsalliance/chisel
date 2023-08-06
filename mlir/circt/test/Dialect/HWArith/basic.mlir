// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @test1() {
hw.module @test1() {
  // CHECK: %0 = hwarith.constant 0 : ui1
  // CHECK: %1 = hwarith.constant 1 : ui1
  // CHECK: %2 = hwarith.constant 2 : ui2
  // CHECK: %3 = hwarith.constant 22 : ui5
  // CHECK: %c7_i5 = hw.constant 7 : i5
  // CHECK: %4 = hwarith.cast %c7_i5 : (i5) -> ui3
  %0 = hwarith.constant 0 : ui1
  %1 = hwarith.constant 1 : ui1
  %2 = hwarith.constant 2 : ui2
  %3 = hwarith.constant 22 : ui5
  %c7_i5 = hw.constant 7 : i5
  %4 = hwarith.cast %c7_i5 : (i5) -> ui3

  // CHECK: %5 = hwarith.add %1, %1 : (ui1, ui1) -> ui2
  %5 = hwarith.add %1, %1 : (ui1, ui1) -> ui2
  // CHECK: %6 = hwarith.add %0, %2 : (ui1, ui2) -> ui3
  %6 = hwarith.add %0, %2 : (ui1, ui2) -> ui3
  // CHECK: %7 = hwarith.mul %2, %3 : (ui2, ui5) -> ui7
  %7 = hwarith.mul %2, %3 : (ui2, ui5) -> ui7
  // CHECK: %8 = hwarith.add %7, %1 : (ui7, ui1) -> ui8
  %8 = hwarith.add %7, %1 : (ui7, ui1) -> ui8
  // CHECK: %9 = hwarith.sub %8, %2 : (ui8, ui2) -> si9
  %9 = hwarith.sub %8, %2 : (ui8, ui2) -> si9
  // CHECK: %10 = hwarith.div %9, %1 : (si9, ui1) -> si9
  %10 = hwarith.div %9, %1 : (si9, ui1) -> si9
  // CHECK: %11 = hwarith.add %10, %6 : (si9, ui3) -> si10
  %11 = hwarith.add %10, %6 : (si9, ui3) -> si10
  // CHECK: %12 = hwarith.cast %11 : (si10) -> i9
  %12 = hwarith.cast %11 : (si10) -> i9
  // CHECK: %13 = hwarith.icmp eq %5, %10 : ui2, si9
  %13 = hwarith.icmp eq %5, %10 : ui2, si9
  // CHECK: %14 = hwarith.cast %13 : (ui1) -> i1
  %14 = hwarith.cast %13 : (ui1) -> i1
}

// RUN: circt-opt %s -canonicalize | FileCheck %s

// CHECK: hw.module @test1(%a: ui8) -> (out: ui8) {
// CHECK:   hw.output %a : ui8
hw.module @test1(%a : ui8) -> (out: ui8) {
    %0 = hwarith.cast %a : (ui8) -> (ui8)
    hw.output %0 : ui8
}

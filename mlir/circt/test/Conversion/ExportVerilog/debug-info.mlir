// RUN: circt-opt --export-verilog %s | FileCheck %s

// CHECK-LABEL: module symbols
// CHECK-NEXT: input baz /* #hw<innerSym@bazSym> */
module attributes {circt.loweringOptions="printDebugInfo"} {
hw.module @symbols(%baz: i1 {hw.exportPort = #hw<innerSym@bazSym>}) -> () {
    // CHECK: wire foo /* #hw<innerSym@fooSym> */;
    %foo = sv.wire sym @fooSym : !hw.inout<i1>
    // CHECK: reg bar /* #hw<innerSym@barSym> */;
    %bar = sv.reg sym @barSym : !hw.inout<i1>
}
}

// RUN: circt-opt -test-apply-lowering-options='options=maximumNumberOfTermsPerExpression=4' -export-verilog %s | FileCheck %s -check-prefixes=CHECK,MAX_8
// RUN: circt-opt -test-apply-lowering-options='options=maximumNumberOfTermsPerExpression=2' -export-verilog %s | FileCheck %s -check-prefixes=CHECK,MAX_4

hw.module @Baz(
  %a0: i1, %a1: i1, %a2: i1, %a3: i1,
  %a4: i1, %a5: i1, %a6: i1, %a7: i1,
  %b0: i1, %b1: i1, %b2: i1, %b3: i1,
  %b4: i1, %b5: i1, %b6: i1, %b7: i1
) -> (c: i1) {
  %0 = comb.and %a0, %b0 : i1
  %1 = comb.and %a1, %b1 : i1
  %2 = comb.and %a2, %b2 : i1
  %3 = comb.and %a3, %b3 : i1
  %4 = comb.and %a4, %b4 : i1
  %5 = comb.and %a5, %b5 : i1
  %6 = comb.and %a6, %b6 : i1
  %7 = comb.and %a7, %b7 : i1
  %8 = comb.or %0, %1, %2, %3, %4, %5, %6, %7 : i1
  hw.output %8 : i1
}

// CHECK-LABEL: module Baz

// MAX_8:      wire [[wire_0:.+]] = a0 & b0 | a1 & b1 | a2 & b2 | a3 & b3;
// MAX_8-NEXT: wire [[wire_1:.+]] = a4 & b4 | a5 & b5 | a6 & b6 | a7 & b7;
// MAX_8-NEXT: assign c = [[wire_0]] | [[wire_1]];

// MAX_4:      wire [[wire_0:.+]] = a0 & b0 | a1 & b1;
// MAX_4-NEXT: wire [[wire_1:.+]] = a2 & b2 | a3 & b3;
// MAX_4-NEXT: wire [[wire_2:.+]] = a4 & b4 | a5 & b5;
// MAX_4-NEXT: wire [[wire_3:.+]] = a6 & b6 | a7 & b7;
// MAX_4-NEXT: assign c = [[wire_0]] | [[wire_1]] | [[wire_2]] | [[wire_3]];

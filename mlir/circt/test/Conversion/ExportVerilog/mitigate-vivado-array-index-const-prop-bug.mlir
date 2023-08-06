// RUN: circt-opt --export-verilog %s | FileCheck %s
// RUN: circt-opt --test-apply-lowering-options='options=mitigateVivadoArrayIndexConstPropBug' --export-verilog %s | FileCheck %s --check-prefix=FIXED

// CHECK-LABEL: module Simple(
// FIXED-LABEL: module Simple(
hw.module @Simple(%a: !hw.array<16xi1>, %b : i4) -> (c: i1) {
  // CHECK: assign c = a[b + 4'h1];

  // FIXED:      (* keep = "true" *)
  // FIXED-NEXT: wire [3:0] [[IDX0:.+]] = b + 4'h1;
  // FIXED-NEXT: assign c = a[[[IDX0]]];

  %c1_i4 = hw.constant 1 : i4
  %0 = comb.add %b, %c1_i4 : i4
  %1 = hw.array_get %a[%0] : !hw.array<16xi1>, i4
  hw.output %1 : i1
}
// CHECK: endmodule
// FIXED: endmodule


// CHECK-LABEL: module ExistingWire(
// FIXED-LABEL: module ExistingWire(
hw.module @ExistingWire(%a: !hw.array<16xi1>, %b : i4) -> (c: i1) {
  // CHECK:      wire [3:0] existingWire = b + 4'h3;
  // CHECK-NEXT: assign c = a[existingWire];

  // FIXED:      (* keep = "true" *)
  // FIXED-NEXT: wire [3:0] existingWire = b + 4'h3;
  // FIXED-NEXT: assign c = a[existingWire];

  %c1_i4 = hw.constant 3 : i4
  %0 = comb.add %b, %c1_i4 : i4
  %existingWire = sv.wire : !hw.inout<i4>
  sv.assign %existingWire, %0 : i4
  %1 = sv.read_inout %existingWire : !hw.inout<i4>
  %2 = hw.array_get %a[%1] : !hw.array<16xi1>, i4
  hw.output %2 : i1
}
// CHECK: endmodule
// FIXED: endmodule


// CHECK-LABEL: module ProceduralRegion(
// FIXED-LABEL: module ProceduralRegion(
hw.module @ProceduralRegion(%a: !hw.array<16xi1>, %b : i4) {
  // CHECK:      magic(a[b + 4'h1]);

  // FIXED:      initial begin
  // FIXED-NEXT:   (* keep = "true" *)
  // FIXED-NEXT:   automatic logic [3:0] [[IDX0:.+]] = b + 4'h1;
  // FIXED-NEXT:   magic(a[[[IDX0]]]);
  // FIXED-NEXT: end

  %c1_i4 = hw.constant 1 : i4
  %0 = comb.add %b, %c1_i4 : i4
  sv.initial {
    %1 = hw.array_get %a[%0] : !hw.array<16xi1>, i4
    sv.verbatim "magic({{0}});" (%1): i1
  }
}
// CHECK: endmodule
// FIXED: endmodule

// RUN: circt-opt --split-input-file --export-verilog %s | FileCheck %s --check-prefix=DEFAULT
// RUN: circt-opt --test-apply-lowering-options='options=' --split-input-file --export-verilog %s | FileCheck %s --check-prefix=CLEAR
// RUN: circt-opt --test-apply-lowering-options='options=noAlwaysComb' --split-input-file --export-verilog %s | FileCheck %s --check-prefix=NOALWAYSCOMB

hw.module @test() {
  sv.alwayscomb {
  }
}

// DEFAULT: always_comb begin
// DEFAULT: end // always_comb

// CLEAR: always_comb begin
// CLEAR: end // always_comb

// NOALWAYSCOMB: always @(*) begin
// NOALWAYSCOMB: end // always @(*)

// -----

module attributes {circt.loweringOptions = "noAlwaysComb"} {
hw.module @test() {
  sv.alwayscomb {
  }
}
}

// DEFAULT: always @(*) begin
// DEFAULT: end // always @(*)

// CLEAR: always_comb begin
// CLEAR: end // always_comb

// NOALWAYSCOMB: always @(*) begin
// NOALWAYSCOMB: end // always @(*)

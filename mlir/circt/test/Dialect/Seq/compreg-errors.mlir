// RUN: circt-opt %s -verify-diagnostics --split-input-file

hw.module @top(%clk: i1, %rst: i1, %i: i32) {
  // expected-error@+1 {{'seq.compreg' expected resetValue operand}}
  seq.compreg %i, %clk, %rst : i32
}

// -----

hw.module @top(%clk: i1, %rst: i1, %i: i32) {
  // expected-error@+1 {{'seq.compreg' expected clock operand}}
  seq.compreg %i : i32
}

// -----


hw.module @top(%clk: i1, %rst: i1, %i: i32) {
  // expected-error@+1 {{'seq.compreg' expected operands}}
  seq.compreg : i32
}

// -----

hw.module @top(%clk: i1, %rst: i1, %i: i32) {
  %rv = hw.constant 0 : i32
  // expected-error@+1 {{'seq.compreg' too many operands}}
  seq.compreg %i, %clk, %rst, %rv, %rv : i32
}

// -----
hw.module @top_ce(%clk: i1, %rst: i1, %ce: i1, %i: i32) {
  // expected-error@+1 {{'seq.compreg.ce' expected resetValue operand}}
  %r0 = seq.compreg.ce %i, %clk, %ce, %rst : i32
}

// -----

hw.module @top_ce(%clk: i1, %rst: i1, %ce: i1, %i: i32) {
  // expected-error@+1 {{'seq.compreg.ce' expected clock enable}}
  %r0 = seq.compreg.ce %i, %clk : i32
}

// -----

hw.module @top_ce(%clk: i1, %rst: i1, %ce: i1, %i: i32) {
  %rv = hw.constant 0 : i32
  // expected-error@+1 {{'seq.compreg.ce' too many operands}}
  %r0 = seq.compreg.ce %i, %clk, %ce, %rst, %rv, %rv : i32
}

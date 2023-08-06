// RUN: circt-opt %s --arc-split-loops --verify-diagnostics --split-input-file

hw.module @UnbreakableLoop(%a: i4) -> (x: i4) {
  // expected-error @below {{loop splitting did not eliminate all loops; loop detected}}
  // expected-note @below {{through operand 1 here:}}
  %0, %1 = arc.state @UnbreakableLoopArc(%a, %0) lat 0 : (i4, i4) -> (i4, i4)
  hw.output %1 : i4
}

arc.define @UnbreakableLoopArc(%arg0: i4, %arg1: i4) -> (i4, i4) {
  %true = hw.constant true
  %0:2 = scf.if %true -> (i4, i4) {
    scf.yield %arg0, %arg1 : i4, i4
  } else {
    scf.yield %arg1, %arg0 : i4, i4
  }
  arc.output %0#0, %0#1 : i4, i4
}

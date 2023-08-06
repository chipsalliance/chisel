// RUN: circt-opt %s -llhd-process-lowering -split-input-file -verify-diagnostics

// Check wait with observing probed signals
llhd.proc @prbAndWaitNotObserved(%arg0 : !llhd.sig<i64>) -> () {
  cf.br ^bb1
^bb1:
  %0 = llhd.prb %arg0 : !llhd.sig<i64>
  // expected-error @+1 {{during process-lowering: the wait terminator is required to have all probed signals as arguments}}
  llhd.wait ^bb1
}

// -----

// Check that block arguments for the second block are not allowed.
// expected-error @+1 {{during process-lowering: the second block (containing the llhd.wait) is not allowed to have arguments}}
llhd.proc @blockArgumentsNotAllowed(%arg0 : !llhd.sig<i64>) -> () {
  cf.br ^bb1(%arg0 : !llhd.sig<i64>)
^bb1(%a : !llhd.sig<i64>):
  llhd.wait ^bb1(%a : !llhd.sig<i64>)
}

// -----

// Check that the entry block is terminated by a cf.br terminator.
// expected-error @+1 {{during process-lowering: the first block has to be terminated by a cf.br operation}}
llhd.proc @entryBlockMustHaveBrTerminator() -> () {
  llhd.wait ^bb1
^bb1:
  llhd.wait ^bb1
}

// -----

// Check that there is no optional time operand in the wait terminator.
llhd.proc @noOptionalTime() -> () {
  cf.br ^bb1
^bb1:
  %time = llhd.constant_time #llhd.time<0ns, 0d, 0e>
  // expected-error @+1 {{during process-lowering: llhd.wait terminators with optional time argument cannot be lowered to structural LLHD}}
  llhd.wait for %time, ^bb1
}

// -----

// Check that if there are two blocks, the second one is terminated by a wait terminator.
// expected-error @+1 {{during process-lowering: the second block must be terminated by llhd.wait}}
llhd.proc @secondBlockTerminatedByWait() -> () {
  cf.br ^bb1
^bb1:
  llhd.halt
}

// -----

// Check that there are not more than two blocks.
// expected-error @+1 {{process-lowering only supports processes with either one basic block terminated by a llhd.halt operation or two basic blocks where the first one contains a cf.br terminator and the second one is terminated by a llhd.wait operation}}
llhd.proc @moreThanTwoBlocksNotAllowed() -> () {
  cf.br ^bb1
^bb1:
  cf.br ^bb2
^bb2:
  llhd.wait ^bb1
}

// -----

llhd.proc @muxedSignal(%arg0 : !llhd.sig<i64>, %arg1 : !llhd.sig<i64>, %arg2 : !llhd.sig<i1>) -> () {
  cf.br ^bb1
^bb1:
  %cond = llhd.prb %arg2 : !llhd.sig<i1>
  %sig = comb.mux %cond, %arg0, %arg1 : !llhd.sig<i64>
  %0 = llhd.prb %sig : !llhd.sig<i64>
  // expected-error @+1 {{during process-lowering: the wait terminator is required to have all probed signals as arguments}}
  llhd.wait (%arg0, %arg2 : !llhd.sig<i64>, !llhd.sig<i1>), ^bb1
}

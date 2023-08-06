// RUN: circt-opt -lower-std-to-handshake %s --canonicalize --split-input-file --verify-diagnostics

// expected-error @+1 {{expected a merge block to have two predecessors.}}
func.func @missingMergeBlocks(%arg0: i1) {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  cf.cond_br %arg0, ^bb3, ^bb4
^bb2:  // pred: ^bb0
  cf.br ^bb4
^bb3:  // pred: ^bb1
// expected-note @+1 {{This branch jumps to the illegal block}}
  cf.br ^bb4
^bb4:  // 3 preds: ^bb1, ^bb2, ^bb3
  return
}

// -----

// expected-error @+1 {{expected only reducible control flow.}}
func.func @irreducibleCFG(%cond: i1) {
  cf.cond_br %cond, ^1, ^2
^1:
  cf.cond_br %cond, ^3, ^4
^2:
  cf.br ^4
^3:
  cf.br ^5
^4:
// expected-note @+1 {{This branch is involved in the irreducible control flow}}
  cf.br ^5
^5:
  return
}

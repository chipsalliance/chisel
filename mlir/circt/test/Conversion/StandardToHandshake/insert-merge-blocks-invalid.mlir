// RUN: circt-opt --insert-merge-blocks %s --split-input-file --verify-diagnostics

// expected-error @+2 {{failed to legalize operation}}
// expected-error @+1 {{irregular control flow is not yet supported}}
func.func @irregular_cfg_two_preds(%cond: i1) {
  cf.cond_br %cond, ^1, ^2
^1:
  cf.cond_br %cond, ^3, ^4
^2:
  cf.br ^4
^3:
  cf.br ^5
^4:
  cf.br ^5
^5:
  return
}

// -----

// expected-error @+2 {{failed to legalize operation}}
// expected-error @+1 {{irregular control flow is not yet supported}}
func.func @irregular_cfg_three_preds(%cond: i1) {
  cf.cond_br %cond, ^1, ^2
^1:
  cf.cond_br %cond, ^3, ^4
^2:
  cf.cond_br %cond, ^4, ^5
^3:
  cf.br ^6
^4:
  cf.br ^6
^5:
  cf.br ^6
^6:
  return
}

// -----

// expected-error @+2 {{failed to legalize operation}}
// expected-error @+1 {{multiple exit blocks are not yet supported}}
func.func @multiple_exit_blocks(%cond: i1) {
  cf.br ^1
^1:
  cf.cond_br %cond, ^2, ^4
^2:
  cf.cond_br %cond, ^1, ^3
^3:
  cf.br ^4
^4:
  return
}

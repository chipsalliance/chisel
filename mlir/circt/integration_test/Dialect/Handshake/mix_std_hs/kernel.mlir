func.func @compute(%val: i64) -> i64 {
  %c1 = arith.constant 1 : i64
  %shr1 = arith.shrui %val, %c1 : i64
  %cond0 = arith.trunci %shr1 : i64 to i1
  cf.cond_br %cond0, ^bb1, ^bb5(%val: i64)
^bb1:
  %c2 = arith.constant 2 : i64
  %shr2 = arith.shrui %val, %c2 : i64
  %cond1 = arith.trunci %shr2 : i64 to i1
  cf.cond_br %cond1, ^bb2, ^bb3
^bb2:
  %t0 = arith.addi %c1, %val : i64
  %t1 = arith.addi %c1, %t0 : i64
  %t2 = arith.addi %c1, %t1 : i64
  %t3 = arith.addi %c1, %t2 : i64
  cf.br ^bb4(%t3: i64)
^bb3:
  %c10 = arith.constant 10 : i64
  %t4 = arith.muli %c10, %val : i64
  cf.br ^bb4(%t4: i64)
^bb4(%t5: i64):
  cf.br ^bb5(%t5: i64)
^bb5(%res: i64):
  return %res: i64
}

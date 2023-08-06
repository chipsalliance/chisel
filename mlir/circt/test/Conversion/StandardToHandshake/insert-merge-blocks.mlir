// RUN: circt-opt --insert-merge-blocks %s --split-input-file | FileCheck %s


// CHECK-LABEL: func.func @noWorkNeeded(
// CHECK-SAME:    %[[ARG1:.*]]: i1) {
// CHECK-NEXT:   cf.cond_br %[[ARG1]], ^bb1, ^bb2
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   cf.br ^bb3
// CHECK-NEXT: ^bb2:
// CHECK-NEXT:   cf.br ^bb3
// CHECK-NEXT: ^bb3:
// CHECK-NEXT:   return
// CHECK-NEXT: }

func.func @noWorkNeeded(%cond: i1) {
  cf.cond_br %cond, ^1, ^2
^1:
  cf.br ^3
^2:
  cf.br ^3
^3:
  return
}

// -----

// CHECK-LABEL: func.func @blockWith3Preds(
// CHECK-SAME:    %[[ARG:.*]]: i1) {
// CHECK-NEXT:   cf.cond_br %[[ARG]], ^bb1, ^bb4
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   cf.cond_br %[[ARG]], ^bb2, ^bb3
// CHECK-NEXT: ^bb2:
// CHECK-NEXT:   cf.br ^bb5
// CHECK-NEXT: ^bb3:
// CHECK-NEXT:   cf.br ^bb5
// CHECK-NEXT: ^bb4:
// CHECK-NEXT:   cf.br ^bb6
// CHECK-NEXT: ^bb5:
// CHECK-NEXT:   cf.br ^bb6
// CHECK-NEXT: ^bb6:
// CHECK-NEXT:   return
// CHECK-NEXT: }

func.func @blockWith3Preds(%cond: i1) {
  cf.cond_br %cond, ^1, ^4
^1:
  cf.cond_br %cond, ^2, ^3
^2:
  cf.br ^end
^3:
  cf.br ^end
^4:
  cf.br ^end
^end:
  return
}

// -----

// CHECK-LABEL: func.func @splitToDirectMerge(
// CHECK-SAME:    %[[ARG:.*]]: i1) {
// CHECK-NEXT:   cf.cond_br %[[ARG]], ^bb1, ^bb2
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   cf.br ^bb2
// CHECK-NEXT: ^bb2:
// CHECK-NEXT:   return
// CHECK-NEXT: }

func.func @splitToDirectMerge(%cond: i1) {
  cf.cond_br %cond, ^1, ^2
^1:
  cf.br ^2
^2:
  return
}

// -----

// CHECK-LABEL: func.func @splitTo3Merge(
// CHECK-SAME:    %[[ARG:.*]]: i1) {
// CHECK-NEXT:   cf.cond_br %[[ARG]], ^bb1, ^bb4
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   cf.cond_br %[[ARG]], ^bb2, ^bb3
// CHECK-NEXT: ^bb2:
// CHECK-NEXT:   cf.br ^bb3
// CHECK-NEXT: ^bb3:
// CHECK-NEXT:   cf.br ^bb4
// CHECK-NEXT: ^bb4:
// CHECK-NEXT:   return
// CHECK-NEXT: }

func.func @splitTo3Merge(%cond: i1) {
  cf.cond_br %cond, ^1, ^3
^1:
  cf.cond_br %cond, ^2, ^3
^2:
  cf.br ^3
^3:
  return
}

// -----

// CHECK-LABEL: func.func @multiple_blocks_needed(
// CHECK-SAME:        %[[ARG:.*]]: i1) {
// CHECK-NEXT:    cf.cond_br %[[ARG]], ^bb1, ^bb4
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    cf.cond_br %[[ARG]], ^bb2, ^bb3
// CHECK-NEXT:  ^bb2:  // pred: ^bb1
// CHECK-NEXT:    cf.br ^bb3
// CHECK-NEXT:  ^bb3:  // 2 preds: ^bb1, ^bb2
// CHECK-NEXT:    cf.br ^bb4
// CHECK-NEXT:  ^bb4:  // 2 preds: ^bb0, ^bb3
// CHECK-NEXT:    cf.cond_br %[[ARG]], ^bb5, ^bb8
// CHECK-NEXT:  ^bb5:  // pred: ^bb4
// CHECK-NEXT:    cf.cond_br %[[ARG]], ^bb6, ^bb7
// CHECK-NEXT:  ^bb6:  // pred: ^bb5
// CHECK-NEXT:    cf.br ^bb7
// CHECK-NEXT:  ^bb7:  // 2 preds: ^bb5, ^bb6
// CHECK-NEXT:    cf.br ^bb8
// CHECK-NEXT:  ^bb8:  // 2 preds: ^bb4, ^bb7
// CHECK-NEXT:    return
// CHECK-NEXT:  }

func.func @multiple_blocks_needed(%cond: i1) {
  cf.cond_br %cond, ^1, ^3
^1:
  cf.cond_br %cond, ^2, ^3
^2:
  cf.br ^3
^3:
  cf.cond_br %cond, ^4, ^end
^4:
  cf.cond_br %cond, ^5, ^end
^5:
  cf.br ^end
^end:
  return
}

// -----

// CHECK-LABEL: func.func @simple_loop(%{{.*}}: i64) {
// CHECK-NEXT:    %{{.*}} = arith.constant 1 : i64
// CHECK-NEXT:    cf.br ^[[BB1:.*]](%{{.*}} : i64)
// CHECK-NEXT:  ^[[BB1]](%{{.*}}: i64):  // 2 preds: ^{{.*}}, ^[[BB2:.*]]
// CHECK-NEXT:    %1 = arith.cmpi eq, %{{.*}}, %{{.*}} : i64
// CHECK-NEXT:    cf.cond_br %{{.*}}, ^[[BB3:.*]], ^[[BB2]]
// CHECK-NEXT:  ^[[BB2]]:  // pred: ^[[BB1]]
// CHECK-NEXT:    %{{.*}} = arith.constant 1 : i64
// CHECK-NEXT:    %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i64
// CHECK-NEXT:    cf.br ^[[BB1]](%{{.*}} : i64)
// CHECK-NEXT:  ^[[BB3]]:  // pred: ^[[BB1]]
// CHECK-NEXT:    return
// CHECK-NEXT:  }

func.func @simple_loop(%n: i64) {
  %c0 = arith.constant 1 : i64
  cf.br ^0(%c0 : i64)
^0(%i: i64):
  %cond = arith.cmpi eq, %i, %n : i64
  cf.cond_br %cond, ^2, ^1
^1:
  %c1 = arith.constant 1 : i64
  %ni = arith.addi %i, %c1 : i64
  cf.br ^0(%ni: i64)
^2:
  return
}

// -----

// CHECK-LABEL:  func.func @blockWith3PredsAndLoop(
// CHECK-SAME:        %[[ARG:.*]]: i1) {
// CHECK-NEXT:     cf.cond_br %[[ARG]], ^bb1, ^bb4
// CHECK-NEXT:   ^bb1:  // pred: ^bb0
// CHECK-NEXT:     cf.cond_br %[[ARG]], ^bb2, ^bb3
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     cf.br ^bb7
// CHECK-NEXT:   ^bb3:  // pred: ^bb1
// CHECK-NEXT:     cf.br ^bb7
// CHECK-NEXT:   ^bb4:  // pred: ^bb0
// CHECK-NEXT:     cf.br ^bb5
// CHECK-NEXT:   ^bb5:  // 2 preds: ^bb4, ^bb6
// CHECK-NEXT:     cf.cond_br %[[ARG]], ^bb8, ^bb6
// CHECK-NEXT:   ^bb6:  // pred: ^bb5
// CHECK-NEXT:     cf.br ^bb5
// CHECK-NEXT:   ^bb7:  // 2 preds: ^bb2, ^bb3
// CHECK-NEXT:     cf.br ^bb8
// CHECK-NEXT:   ^bb8:  // 2 preds: ^bb5, ^bb7
// CHECK-NEXT:     return
// CHECK-NEXT:   }

func.func @blockWith3PredsAndLoop(%cond: i1) {
  cf.cond_br %cond, ^1, ^4
^1:
  cf.cond_br %cond, ^2, ^3
^2:
  cf.br ^end
^3:
  cf.br ^end
^4:
  cf.br ^5
^5:
  cf.cond_br %cond, ^end, ^6
^6:
  cf.br ^5
^end:
  return
}

// -----


// CHECK-LABEL: func.func @otherBlockOrder(
// CHECK-SAME:        %[[ARG:.*]]: i1) {
// CHECK-NEXT:    cf.cond_br %[[ARG]], ^bb1, ^bb4
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    cf.cond_br %[[ARG]], ^bb2, ^bb3
// CHECK-NEXT:  ^bb2:  // pred: ^bb1
// CHECK-NEXT:    cf.br ^bb7
// CHECK-NEXT:  ^bb3:  // pred: ^bb1
// CHECK-NEXT:    cf.br ^bb7
// CHECK-NEXT:  ^bb4:  // pred: ^bb0
// CHECK-NEXT:    cf.br ^bb5
// CHECK-NEXT:  ^bb5:  // 2 preds: ^bb4, ^bb6
// CHECK-NEXT:    cf.br ^bb6
// CHECK-NEXT:  ^bb6:  // pred: ^bb5
// CHECK-NEXT:    cf.cond_br %[[ARG]], ^bb8, ^bb5
// CHECK-NEXT:  ^bb7:  // 2 preds: ^bb2, ^bb3
// CHECK-NEXT:    cf.br ^bb8
// CHECK-NEXT:  ^bb8:  // 2 preds: ^bb6, ^bb7
// CHECK-NEXT:    return
// CHECK-NEXT:  }

func.func @otherBlockOrder(%cond: i1) {
  cf.cond_br %cond, ^1, ^4
^1:
  cf.cond_br %cond, ^2, ^3
^2:
  cf.br ^end
^3:
  cf.br ^end
^4:
  cf.br ^5
^5:
  cf.br ^6
^6:
  cf.cond_br %cond, ^end, ^5
^end:
  return
}

// -----

// CHECK-LABEL: func.func @multiple_block_args(
// CHECK-SAME:      %[[COND:.*]]: i1,
// CHECK-SAME:      %[[VAL:.*]]: i64) {
// CHECK-NEXT:     cf.cond_br %[[COND]], ^bb1(%[[VAL]] : i64), ^bb4(%[[VAL]], %arg1 : i64, i64)
// CHECK-NEXT:   ^bb1(%[[VAL1:.*]]: i64):  // pred: ^bb0
// CHECK-NEXT:     cf.cond_br %[[COND]], ^bb2(%[[VAL1]] : i64), ^bb3(%[[VAL1]], %[[VAL1]] : i64, i64)
// CHECK-NEXT:   ^bb2(%{{.*}}: i64):  // pred: ^bb1
// CHECK-NEXT:     cf.br ^bb5
// CHECK-NEXT:   ^bb3(%{{.*}}: i64, %{{.*}}: i64):  // pred: ^bb1
// CHECK-NEXT:     cf.br ^bb5
// CHECK-NEXT:   ^bb4(%{{.*}}: i64, %{{.*}}: i64):  // pred: ^bb0
// CHECK-NEXT:     cf.br ^bb6
// CHECK-NEXT:   ^bb5:  // 2 preds: ^bb2, ^bb3
// CHECK-NEXT:     cf.br ^bb6
// CHECK-NEXT:   ^bb6:  // 2 preds: ^bb4, ^bb5
// CHECK-NEXT:     return
// CHECK-NEXT:   }

func.func @multiple_block_args(%cond: i1, %val: i64) {
  cf.cond_br %cond, ^1(%val: i64), ^4(%val, %val: i64, i64)
^1(%val1: i64):
  cf.cond_br %cond, ^2(%val1: i64), ^3(%val1, %val1: i64, i64)
^2(%val2: i64):
  cf.br ^end
^3(%val3: i64, %val3_1: i64):
  cf.br ^end
^4(%val4: i64, %val4_1: i64):
  cf.br ^end
^end:
  return
}

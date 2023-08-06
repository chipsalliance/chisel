// RUN: circt-opt %s --canonicalize --cse --canonicalize | FileCheck %s
// XFAIL: *
// Waiting on: https://github.com/llvm/llvm-project/issues/64280

// CHECK-LABEL:   func.func @staggeredJoin1(
// CHECK-SAME:                    %[[VAL_0:.*]]: !dc.token,
// CHECK-SAME:                    %[[VAL_1:.*]]: !dc.token) -> !dc.token {
// CHECK:           %[[VAL_2:.*]] = dc.join %[[VAL_0]], %[[VAL_1]]
// CHECK:           return %[[VAL_2]] : !dc.token
// CHECK:         }
func.func @staggeredJoin1(%a: !dc.token, %b : !dc.token) -> (!dc.token) {
    %0 = dc.join %a, %b
    %1 = dc.join %0
    return %1 : !dc.token
}

// CHECK-LABEL:   func.func @staggeredJoin2(
// CHECK-SAME:           %[[VAL_0:.*]]: !dc.token, %[[VAL_1:.*]]: !dc.token, %[[VAL_2:.*]]: !dc.token, %[[VAL_3:.*]]: !dc.token) -> !dc.token {
// CHECK:           %[[VAL_4:.*]] = dc.join %[[VAL_2]], %[[VAL_3]], %[[VAL_0]], %[[VAL_1]]
// CHECK:           return %[[VAL_4]] : !dc.token
// CHECK:         }
func.func @staggeredJoin2(%a: !dc.token, %b : !dc.token, %c : !dc.token, %d : !dc.token) -> (!dc.token) {
    %0 = dc.join %a, %b
    %1 = dc.join %c, %0, %d
    return %1 : !dc.token
}

// CHECK-LABEL:   func.func @redundantJoinOperands(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !dc.token,
// CHECK-SAME:                                     %[[VAL_1:.*]]: !dc.token) -> !dc.token {
// CHECK:           %[[VAL_2:.*]] = dc.join %[[VAL_0]], %[[VAL_1]]
// CHECK:           return %[[VAL_2]] : !dc.token
// CHECK:         }
func.func @redundantJoinOperands(%a: !dc.token, %b : !dc.token) -> (!dc.token) {
    %0 = dc.join %a, %b, %b
    return %0 : !dc.token
}

// CHECK-LABEL:   func.func @fork(
// CHECK-SAME:                    %[[VAL_0:.*]]: !dc.token) -> !dc.token {
// CHECK:           return %[[VAL_0]] : !dc.token
// CHECK:         }
func.func @fork(%a: !dc.token) -> (!dc.token) {
    %0 = dc.fork [1] %a
    return %0 : !dc.token
}

// CHECK-LABEL:   func.func @packUnpack(
// CHECK-SAME:                          %[[VAL_0:.*]]: !dc.token,
// CHECK-SAME:                          %[[VAL_1:.*]]: i32) -> (!dc.token, i32) {
// CHECK:           return %[[VAL_0]], %[[VAL_1]] : !dc.token, i32
// CHECK:         }
func.func @packUnpack(%a: !dc.token, %b : i32) -> (!dc.token, i32) {
    %0 = dc.pack %a, %b : i32
    %1:2 = dc.unpack %0 : !dc.value<i32>
    return %1#0, %1#1 : !dc.token, i32
}

// CHECK-LABEL:   func.func @redundantPack(
// CHECK-SAME:                             %[[VAL_0:.*]]: !dc.token,
// CHECK-SAME:                             %[[VAL_1:.*]]: i32) -> !dc.token {
// CHECK:           return %[[VAL_0]] : !dc.token
// CHECK:         }
func.func @redundantPack(%a: !dc.token, %b : i32) -> (!dc.token) {
    %0 = dc.pack %a, %b : i32 
    %1:2 = dc.unpack %0 : !dc.value<i32>
    return %1#0 : !dc.token
}

// CHECK-LABEL:   func.func @csePack(
// CHECK-SAME:                       %[[VAL_0:.*]]: !dc.token,
// CHECK-SAME:                       %[[VAL_1:.*]]: i32) -> (!dc.value<i32>, !dc.value<i32>) {
// CHECK:           %[[VAL_3:.*]] = dc.pack %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           return %[[VAL_3]], %[[VAL_3]] : !dc.value<i32>, !dc.value<i32>
// CHECK:         }
func.func @csePack(%a: !dc.token, %b : i32) -> (!dc.value<i32>, !dc.value<i32>) {
    %0 = dc.pack %a, %b : i32
    %1 = dc.pack %a, %b : i32
    return %0, %1 : !dc.value<i32>, !dc.value<i32>
}


// CHECK-LABEL:   func.func @forkToFork(
// CHECK-SAME:                          %[[VAL_0:.*]]: !dc.token) -> (!dc.token, !dc.token, !dc.token) {
// CHECK:           %[[VAL_1:.*]]:3 = dc.fork [3] %[[VAL_0]]
// CHECK:           return %[[VAL_1]]#1, %[[VAL_1]]#1, %[[VAL_1]]#2 : !dc.token, !dc.token, !dc.token
// CHECK:         }
func.func @forkToFork(%a: !dc.token) -> (!dc.token, !dc.token, !dc.token) {
    %0, %1 = dc.fork [2] %a
    %2, %3 = dc.fork [2] %0
    return %1, %2, %3 : !dc.token, !dc.token, !dc.token
}

// CHECK-LABEL:   func.func @forkToFork2(
// CHECK-SAME:                           %[[VAL_0:.*]]: !dc.token) -> (!dc.token, !dc.token, !dc.token) {
// CHECK:           %[[VAL_1:.*]]:3 = dc.fork [3] %[[VAL_0]]
// CHECK:           return %[[VAL_1]]#0, %[[VAL_1]]#1, %[[VAL_1]]#2 : !dc.token, !dc.token, !dc.token
// CHECK:         }
func.func @forkToFork2(%a: !dc.token) -> (!dc.token, !dc.token, !dc.token) {
    %0, %1 = dc.fork [2] %a
    %2, %3 = dc.fork [2] %1
    return %0, %2, %3 : !dc.token, !dc.token, !dc.token
}

// CHECK-LABEL:   func.func @merge(
// CHECK-SAME:                     %[[VAL_0:.*]]: !dc.value<i1>) -> !dc.token {
// CHECK:           %[[VAL_1:.*]], %[[VAL_2:.*]] = dc.unpack %[[VAL_0]] : !dc.value<i1>
// CHECK:           return %[[VAL_1]] : !dc.token
// CHECK:         }
func.func @merge(%sel : !dc.value<i1>) -> (!dc.token) {
    // Canonicalize away a merge that is fed by a branch with the same select
    // input.
    %true, %false = dc.branch %sel
    %0 = dc.select %sel, %true, %false
    return %0 : !dc.token
}

// CHECK-LABEL:   func.func @joinOnSource(
// CHECK-SAME:                            %[[VAL_0:.*]]: !dc.token,
// CHECK-SAME:                            %[[VAL_1:.*]]: !dc.token) -> !dc.token {
// CHECK:           %[[VAL_2:.*]] = dc.join %[[VAL_0]], %[[VAL_1]]
// CHECK:           return %[[VAL_2]] : !dc.token
// CHECK:         }
func.func @joinOnSource(%a : !dc.token, %b : !dc.token) -> (!dc.token) {
    %0 = dc.source
    %out = dc.join %a, %0, %b
    return %out : !dc.token
}

// CHECK-LABEL:   func.func @forkOfSource() -> (!dc.token, !dc.token) {
// CHECK:           %[[VAL_0:.*]] = dc.source
// CHECK:           return %[[VAL_0]], %[[VAL_0]] : !dc.token, !dc.token
// CHECK:         }
func.func @forkOfSource() -> (!dc.token, !dc.token) {
    %0 = dc.source
    %1:2 = dc.fork [2] %0
    return %1#0, %1#1 : !dc.token, !dc.token
}

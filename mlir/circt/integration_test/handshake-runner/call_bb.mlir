// RUN: handshake-runner %s | FileCheck %s
// RUN: circt-opt -lower-std-to-handshake -handshake-materialize-forks-sinks %s | handshake-runner | FileCheck %s
// CHECK: 763 2996
module {
  func.func @muladd(%1:index, %2:index, %3:index) -> (index) {
    %i2 = arith.muli %1, %2 : index
    %i3 = arith.addi %3, %i2 : index
	 return %i3 : index
  }

  func.func @main() -> (index, index) {
    %c0 = arith.constant 0 : index
    %c101 = arith.constant 101 : index
    %c102 = arith.constant 102 : index
    %0 = arith.addi %c0, %c0 : index
    %c1 = arith.constant 1 : index
    %1 = arith.addi %0, %c102 : index
    %c103 = arith.constant 103 : index
    %c104 = arith.constant 104 : index
    %c105 = arith.constant 105 : index
    %c106 = arith.constant 106 : index
    %c107 = arith.constant 107 : index
    %c108 = arith.constant 108 : index
    %c109 = arith.constant 109 : index
    %c2 = arith.constant 2 : index
  	%3 = call @muladd(%c104, %c2, %c103) : (index, index, index) -> index
    %c3 = arith.constant 3 : index
    %4 = arith.muli %c105, %c3 : index
    %5 = arith.addi %3, %4 : index
    %c4 = arith.constant 4 : index
    %6 = arith.muli %c106, %c4 : index
    %7 = arith.addi %5, %6 : index
    %c5 = arith.constant 5 : index
    %8 = arith.muli %c107, %c5 : index
    %9 = arith.addi %7, %8 : index
    %c6 = arith.constant 6 : index
    %10 = arith.muli %c108, %c6 : index
    %11 = arith.addi %9, %10 : index
    %c7 = arith.constant 7 : index
    %12 = arith.muli %c109, %c7 : index
    %13 = arith.addi %11, %12 : index
    return %12, %13 : index, index
  }
}

// RUN: circt-opt -lower-std-to-handshake %s -split-input-file -verify-diagnostics

func.func @multidim() -> i32 {
  // expected-error @+1 {{memref's must be both statically sized and unidimensional.}}
  %0 = memref.alloc() : memref<2x2xi32>
  %idx = arith.constant 0 : index
  %1 = memref.load %0[%idx, %idx] : memref<2x2xi32>
  return %1 : i32
}

// -----

func.func @dynsize(%dyn : index) -> i32{
  // expected-error @+1 {{memref's must be both statically sized and unidimensional.}}
  %0 = memref.alloc(%dyn) : memref<?xi32>
  %idx = arith.constant 0 : index
  %1 = memref.load %0[%idx] : memref<?xi32>
  return %1 : i32
}

// -----

func.func @singleton() -> (){
  // expected-error @+1 {{memref's must be both statically sized and unidimensional.}}
  %0 = memref.alloc() : memref<i32>
  %1 = memref.load %0[] : memref<i32>
  return
}

// -----

func.func @non_canon_loop(%arg0 : memref<100xi32>, %arg1 : i32) -> i32 {
  // expected-error @below {{expected cmerges to have two operands}}
  %c0_i32 = arith.constant 0 : i32
  %c100 = arith.constant 100 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c = arith.cmpi slt, %arg1, %c0_i32 : i32
  cf.cond_br %c, ^bb1(%c0 : index) , ^bbx
^bbx:
  // Jump directly to the loop body, skipping the header
  cf.br ^bb2(%c0 : index)
^bb1(%0: index):  // Actual loop header. 2 preds: ^bb0, ^bb2
  %1 = arith.cmpi slt, %0, %c100 : index
  cf.cond_br %1, ^bb2(%c0 : index), ^bb3
^bb2(%i : index):  // Also a loop header, due to ^bbx pred: ^bb1, ^bbx
  %2 = arith.index_cast %i : index to i32
  memref.store %2, %arg0[%i] : memref<100xi32>
  %3 = arith.addi %i, %c1 : index
  cf.br ^bb1(%3 : index)
^bb3:  // pred: ^bb1
  return %c0_i32 : i32
}

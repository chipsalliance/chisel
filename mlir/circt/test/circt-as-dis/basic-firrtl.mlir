// RUN: circt-as %s -o - | circt-opt | FileCheck -strict-whitespace %s
// RUN: circt-opt %s -emit-bytecode | circt-dis | FileCheck -strict-whitespace  %s
// RUN: circt-as %s -o - | circt-dis | FileCheck -strict-whitespace  %s

firrtl.circuit "Top" {
  firrtl.module @Top(in %in : !firrtl.uint<8>,
                     out %out : !firrtl.uint<8>) {
    firrtl.strictconnect %out, %in : !firrtl.uint<8>
  }
}

// CHECK-LABEL: firrtl.circuit "Top" {
// CHECK-NEXT:    firrtl.module @Top(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
// CHECK-NEXT:      firrtl.strictconnect %out, %in : !firrtl.uint<8>
// CHECK-NEXT:    }
// CHECK-NEXT:  }

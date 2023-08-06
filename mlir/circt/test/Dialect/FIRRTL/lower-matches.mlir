// RUN: circt-opt --pass-pipeline="builtin.module(firrtl.circuit(firrtl.module(firrtl-lower-matches)))" %s | FileCheck %s

firrtl.circuit "LowerMatches" {

// CHECK-LABEL: firrtl.module @EmptyEnum
firrtl.module @EmptyEnum(in %enum : !firrtl.enum<>) {
  firrtl.match %enum : !firrtl.enum<> {
  }
// CHECK-NEXT: }
}

// CHECK-LABEL: firrtl.module @OneVariant
firrtl.module @OneVariant(in %enum : !firrtl.enum<a: uint<8>>, out %out : !firrtl.uint<8>) {
  // CHECK: %0 = firrtl.subtag %enum[a] : !firrtl.enum<a: uint<8>>
  // CHECK: firrtl.strictconnect %out, %0 : !firrtl.uint<8>
  firrtl.match %enum : !firrtl.enum<a: uint<8>> {
    case a(%arg0) {
      firrtl.strictconnect %out, %arg0 : !firrtl.uint<8>
    }
  }
}

// CHECK-LABEL: firrtl.module @LowerMatches
firrtl.module @LowerMatches(in %enum : !firrtl.enum<a: uint<8>, b: uint<8>, c: uint<8>>, out %out : !firrtl.uint<8>) {

   // CHECK-NEXT: %0 = firrtl.istag %enum a : !firrtl.enum<a: uint<8>, b: uint<8>, c: uint<8>>
   // CHECK-NEXT: firrtl.when %0 : !firrtl.uint<1> {
   // CHECK-NEXT:   %1 = firrtl.subtag %enum[a] : !firrtl.enum<a: uint<8>, b: uint<8>, c: uint<8>>
   // CHECK-NEXT:   firrtl.strictconnect %out, %1 : !firrtl.uint<8>
   // CHECK-NEXT: } else {
   // CHECK-NEXT:   %1 = firrtl.istag %enum b : !firrtl.enum<a: uint<8>, b: uint<8>, c: uint<8>>
   // CHECK-NEXT:   firrtl.when %1 : !firrtl.uint<1> {
   // CHECK-NEXT:     %2 = firrtl.subtag %enum[b] : !firrtl.enum<a: uint<8>, b: uint<8>, c: uint<8>>
   // CHECK-NEXT:     firrtl.strictconnect %out, %2 : !firrtl.uint<8>
   // CHECK-NEXT:   } else {
   // CHECK-NEXT:     %2 = firrtl.subtag %enum[c] : !firrtl.enum<a: uint<8>, b: uint<8>, c: uint<8>>
   // CHECK-NEXT:     firrtl.strictconnect %out, %2 : !firrtl.uint<8>
   // CHECK-NEXT:   }
   // CHECK-NEXT: }
  firrtl.match %enum : !firrtl.enum<a: uint<8>, b: uint<8>, c: uint<8>> {
    case a(%arg0) {
      firrtl.strictconnect %out, %arg0 : !firrtl.uint<8>
    }
    case b(%arg0) {
      firrtl.strictconnect %out, %arg0 : !firrtl.uint<8>
    }
    case c(%arg0) {
      firrtl.strictconnect %out, %arg0 : !firrtl.uint<8>
    }
  }

}

// CHECK-LABEL: firrtl.module @ConstLowerMatches
firrtl.module @ConstLowerMatches(in %enum : !firrtl.const.enum<a: uint<8>, b: uint<8>, c: uint<8>>, out %out : !firrtl.const.uint<8>) {

   // CHECK-NEXT: %0 = firrtl.istag %enum a : !firrtl.const.enum<a: uint<8>, b: uint<8>, c: uint<8>>
   // CHECK-NEXT: firrtl.when %0 : !firrtl.const.uint<1> {
   // CHECK-NEXT:   %1 = firrtl.subtag %enum[a] : !firrtl.const.enum<a: uint<8>, b: uint<8>, c: uint<8>>
   // CHECK-NEXT:   firrtl.strictconnect %out, %1 : !firrtl.const.uint<8>
   // CHECK-NEXT: } else {
   // CHECK-NEXT:   %1 = firrtl.istag %enum b : !firrtl.const.enum<a: uint<8>, b: uint<8>, c: uint<8>>
   // CHECK-NEXT:   firrtl.when %1 : !firrtl.const.uint<1> {
   // CHECK-NEXT:     %2 = firrtl.subtag %enum[b] : !firrtl.const.enum<a: uint<8>, b: uint<8>, c: uint<8>>
   // CHECK-NEXT:     firrtl.strictconnect %out, %2 : !firrtl.const.uint<8>
   // CHECK-NEXT:   } else {
   // CHECK-NEXT:     %2 = firrtl.subtag %enum[c] : !firrtl.const.enum<a: uint<8>, b: uint<8>, c: uint<8>>
   // CHECK-NEXT:     firrtl.strictconnect %out, %2 : !firrtl.const.uint<8>
   // CHECK-NEXT:   }
   // CHECK-NEXT: }
  firrtl.match %enum : !firrtl.const.enum<a: uint<8>, b: uint<8>, c: uint<8>> {
    case a(%arg0) {
      firrtl.strictconnect %out, %arg0 : !firrtl.const.uint<8>
    }
    case b(%arg0) {
      firrtl.strictconnect %out, %arg0 : !firrtl.const.uint<8>
    }
    case c(%arg0) {
      firrtl.strictconnect %out, %arg0 : !firrtl.const.uint<8>
    }
  }

}
}

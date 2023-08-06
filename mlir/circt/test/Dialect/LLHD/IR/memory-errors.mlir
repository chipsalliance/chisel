// RUN: circt-opt %s -mlir-print-op-generic -split-input-file -verify-diagnostics

// expected-note @+1 {{prior use here}}
func.func @check_illegal_store(%i1Ptr : !llhd.ptr<i1>, %i32Const : i32) {
  // expected-error @+1 {{use of value '%i32Const' expects different type than prior uses: 'i1' vs 'i32'}}
  llhd.store %i1Ptr, %i32Const : !llhd.ptr<i1>

  return
}

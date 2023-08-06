// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "firrtl.module @Basic" --keep-best=0 --include memory-stubber | FileCheck %s

firrtl.circuit "Basic"   {
  // CHECK-LABEL: @Basic
  firrtl.module @Basic() {
    %memory_r = firrtl.mem Undefined  {depth = 16 : i64, name = "memory", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
    // CHECK: %memory_r = firrtl.wire : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
    // CHECK: [[MEM_ADDR:%.+]] = firrtl.subfield %memory_r[addr]
    // CHECK: [[MEM_EN:%.+]] = firrtl.subfield %memory_r[en]
    // CHECK: [[XOR:%.+]] = firrtl.xor [[MEM_ADDR]], [[MEM_EN]]
    // CHECK: firrtl.connect {{%.+}}, [[XOR]]
  }

  // CHECK-LABEL: @BundleWithSignedInt
  firrtl.module @BundleWithSignedInt() {
    %memory_r, %memory_w = firrtl.mem Undefined  {depth = 16 : i64, name = "memory", portNames = ["r", "w"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: sint<8>>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<a: uint<8>, b: sint<8>>, mask: bundle<a: uint<1>, b: uint<1>>>
    // CHECK: [[UINT:%.+]] = firrtl.asUInt {{%.+}}
    // CHECK: {{%.+}} = firrtl.xor {{%.+}}, [[UINT]]
    // CHECK: [[SINT:%.+]] = firrtl.asSInt {{%.+}}
    // CHECK: firrtl.connect {{%.+}}, [[SINT]]
  }
}

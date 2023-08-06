// RUN: circt-opt -firrtl-print-nla-table %s  2>&1 | FileCheck %s

// CHECK: BarNL: nla_1, nla,
// CHECK: BazNL: nla_1, nla_0, nla,
// CHECK: FooNL: nla_1, nla_0, nla,
// CHECK: FooL:

firrtl.circuit "FooNL"  {
  hw.hierpath private @nla_1 [@FooNL::@baz, @BazNL::@bar, @BarNL]
  hw.hierpath private @nla_0 [@FooNL::@baz, @BazNL]
  hw.hierpath private @nla [@FooNL::@baz, @BazNL::@bar, @BarNL]

  firrtl.module @BarNL() attributes {annotations = [{circt.nonlocal = @nla_1, class = "circt.test", nl = "nl"}]} {
    %w2 = firrtl.wire sym @w2  {annotations = [{circt.fieldID = 5 : i32, circt.nonlocal = @nla, class = "circt.test", nl = "nl2"}]} : !firrtl.bundle<a: uint, b: vector<uint, 4>>
    firrtl.skip
  }
  firrtl.module @BazNL() {
    %w = firrtl.wire sym @w  {annotations = [{circt.nonlocal = @nla_0, class = "circt.test", nl = "nl"}]} : !firrtl.uint
    firrtl.instance bar sym @bar @BarNL()
  }
  firrtl.module @FooNL() {
    firrtl.instance baz sym @baz @BazNL()
  }
  firrtl.module @FooL() {
    %w3 = firrtl.wire  {annotations = [{class = "circt.test", nl = "nl3"}]} : !firrtl.uint
  }
}

// RUN: circt-opt -firrtl-lower-annotations -mlir-print-op-generic -split-input-file %s | FileCheck %s

// A ReferenceTarget/ComponentName pointing at a module/extmodule port should
// work.
firrtl.circuit "Foo" attributes {rawAnnotations = [
  {class = "circt.test", data = "a", target = "~Foo|Bar>bar"},
  {class = "circt.test", data = "b", target = "Foo.Foo.foo"}
]} {
  firrtl.extmodule @Bar(in bar: !firrtl.uint<1>)
  firrtl.module @Foo(in %foo: !firrtl.uint<1>) {
    %bar_bar = firrtl.instance bar @Bar(in bar: !firrtl.uint<1>)
    firrtl.strictconnect %bar_bar, %foo : !firrtl.uint<1>
  }
}

// CHECK-LABEL: "firrtl.extmodule"() ({
// CHECK: }) {
// CHECK-SAME: portAnnotations = {{['[']['[']}}{class = "circt.test", data = "a"}]]
// CHECK-SAME: sym_name = "Bar"

// CHECK-LABEL: "firrtl.module"() ({
// CHECK: }) {
// CHECK-SAME: portAnnotations = {{['[']['[']}}{class = "circt.test", data = "b"}]]
// CHECK-SAME: sym_name = "Foo"

// RUN: split-file %s %t
// RUN: cd %t
// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-blackbox-reader))' Foo.mlir | FileCheck Foo.mlir
// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-blackbox-reader))' NoDUT.mlir | FileCheck NoDUT.mlir

//--- Baz.sv
/* Baz */
//--- Foo.mlir
firrtl.circuit "Foo" attributes {annotations = [
{class = "sifive.enterprise.firrtl.TestBenchDirAnnotation", dirname = "../testbench"},
{class = "sifive.enterprise.firrtl.ExtractCoverageAnnotation", directory = "cover"}
]}
{
  // CHECK-LABEL: firrtl.extmodule @ExtFoo()
  // CHECK-NOT: class = "firrtl.transforms.BlackBoxInlineAnno"
  // CHECK-SAME: class = "firrtl.transforms.BlackBox"
  firrtl.extmodule @ExtFoo() attributes {annotations = [{class = "firrtl.transforms.BlackBoxInlineAnno", name = "hello.v", text = "// world"}]}
  // CHECK-LABEL: firrtl.extmodule @ExtFoo2()
  // CHECK-NOT: class = "firrtl.transforms.BlackBoxInlineAnno"
  firrtl.extmodule @ExtFoo2() attributes {annotations = [{class = "firrtl.transforms.BlackBoxInlineAnno", name = "hello2.v", text = "// world"}, {class = "freechips.rocketchip.annotations.InternalVerifBlackBoxAnnotation"}]}
  // CHECK-LABEL: firrtl.extmodule @ExtFoo3()
  // CHECK-NOT: class = "firrtl.transforms.BlackBoxInlineAnno"
  firrtl.extmodule @ExtFoo3() attributes {annotations = [{class = "firrtl.transforms.BlackBoxInlineAnno", name = "hello3.v", text = "// world"}, {class = "freechips.rocketchip.annotations.InternalVerifBlackBoxAnnotation"}]}
  // CHECK-LABEL: firrtl.module @DUTBlackboxes
  // CHECK-NOT: class = "firrtl.transforms.BlackBoxInlineAnno"
  firrtl.module @DUTBlackboxes() attributes {annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}, {class = "firrtl.transforms.BlackBoxInlineAnno", name = "hello_dut.v", text = "// world"}]} {
      firrtl.instance foo2  @ExtFoo2()
      firrtl.instance bar @Bar()
      firrtl.instance baz @Baz()
  }
  firrtl.extmodule @Bar() attributes {annotations = [{class = "firrtl.transforms.BlackBoxInlineAnno", name = "Bar.v", text = "/* Bar */\0A"}], output_file = #hw.output_file<"bar/">}
  firrtl.extmodule @Baz() attributes {annotations = [{class = "firrtl.transforms.BlackBoxPathAnno", path = "Baz.sv"}], output_file = #hw.output_file<"baz/">}
  firrtl.extmodule @Qux() attributes {annotations = [{class = "firrtl.transforms.BlackBoxInlineAnno", name = "Qux.sv", text = "/* Qux */\0A"}], output_file = #hw.output_file<"qux/NotQux.jpeg">}
  firrtl.module @Foo() {
    firrtl.instance foo @ExtFoo()
    firrtl.instance foo3 @ExtFoo3()
    firrtl.instance dut @DUTBlackboxes()
  }
  // CHECK: sv.verbatim "// world" {output_file = #hw.output_file<"..{{/|\\\\}}testbench{{/|\\\\}}hello.v">}
  // CHECK: sv.verbatim "// world" {output_file = #hw.output_file<"cover{{/|\\\\}}hello2.v">}
  // CHECK: sv.verbatim "// world" {output_file = #hw.output_file<"..{{/|\\\\}}testbench{{/|\\\\}}hello3.v">}
  // CHECK: sv.verbatim "// world" {output_file = #hw.output_file<".{{/|\\\\}}hello_dut.v">}
  // CHECK: sv.verbatim "/* Bar */\0A" {output_file = #hw.output_file<"bar{{/|\\\\}}Bar.v">}
  // CHECK: sv.verbatim "/* Baz */{{(\\0D)?}}\0A" {output_file = #hw.output_file<"baz{{/|\\\\}}Baz.sv">}
  // CHECK: sv.verbatim "/* Qux */\0A" {output_file = #hw.output_file<"qux{{/|\\\\}}NotQux.jpeg">}
  // CHECK: sv.verbatim "..{{/|\\\\}}testbench{{/|\\\\}}hello.v\0A
  // CHECK-SAME:         ..{{/|\\\\}}testbench{{/|\\\\}}hello3.v\0A
  // CHECK-SAME:         hello_dut.v\0A
  // CHECK-SAME:         bar{{/|\\\\}}Bar.v\0A
  // CHECK-SAME:         baz{{/|\\\\}}Baz.sv\0A
  // CHECK-SAME:         cover{{/|\\\\}}hello2.v\0A
  // CHECK-SAME:         qux{{/|\\\\}}NotQux.jpeg"
  // CHECK-SAME: output_file = #hw.output_file<"firrtl_black_box_resource_files.f", excludeFromFileList>
}
//--- NoDUT.mlir
// Check that a TestBenchDirAnnotation has no effect without the presence of a
// MarkDUTAnnotation.
//
// CHECK: firrtl.circuit "NoDUT"
firrtl.circuit "NoDUT" attributes {annotations = [
  {
    class = "sifive.enterprise.firrtl.TestBenchDirAnnotation",
    dirname = "testbench"
  }
]} {
  firrtl.extmodule @NoDUTBlackBox() attributes {annotations = [
  {
    class = "firrtl.transforms.BlackBoxInlineAnno",
    name = "NoDUTBlackBox.sv",
    text = "module NoDUTBlackBox();\nendmodule\n",
    target = "~NoDUT|NoDUTBlackBox"
  }
]}
  firrtl.module @NoDUT() {
    firrtl.instance noDUTBlackBox @NoDUTBlackBox()
  }
  // CHECK:      sv.verbatim "module NoDUTBlackBox()
  // CHECK-SAME:   #hw.output_file<".{{/|\\\\}}NoDUTBlackBox.sv">
  // CHECK:      sv.verbatim "NoDUTBlackBox.sv"
}

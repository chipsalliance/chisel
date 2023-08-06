// RUN: circt-opt -lower-firrtl-to-hw=warn-on-unprocessed-annotations -verify-diagnostics -split-input-file -allow-unregistered-dialect -mlir-disable-threading %s

// The firrtl.circuit should be removed, the main module name moved to an
// attribute on the module.
// CHECK-NOT: firrtl.circuit

// We should get a large header boilerplate.
// CHECK:   sv.ifdef.procedural "RANDOMIZE_GARBAGE_ASSIGN"  {
// CHECK-NEXT:   sv.verbatim "`define RANDOMIZE"
// CHECK-NEXT:  }

firrtl.circuit "OperandTypeIsFIRRTL" {
  firrtl.module @OperandTypeIsFIRRTL() { }
    func.func @Test() {
    // expected-error @+1 {{Found unhandled FIRRTL operation 'firrtl.constant'}}
    %a = firrtl.constant 0 : !firrtl.uint<1>
    return
  }
}

// -----

firrtl.circuit "ResultTypeIsFIRRTL" {
  firrtl.module @ResultTypeIsFIRRTL() { }
  // expected-error @+1 {{fake_op' op found unhandled FIRRTL type}}
  %1 = "fake_op"() : () -> (!firrtl.uint<1>)
}

// -----

firrtl.circuit "RecursiveCheck" {
  firrtl.module @RecursiveCheck() { }
  func.func private @CheckRecursive() {
    // expected-error @+1 {{fake_op' op found unhandled FIRRTL type}}
    %1 = "fake_op"() : () -> (!firrtl.uint<1>)
  }
}

// -----

firrtl.circuit "BlockArgType" {
  firrtl.module @BlockArgType() { }
  // expected-error @+1 {{fake_op' op found unhandled FIRRTL type}}
  "fake_op"() ({
    ^bb0(%a: !firrtl.uint<1>):
      "fake_return"() : () -> ()
    }): () -> ()
}

// -----

firrtl.circuit "unprocessedAnnotations" {
 firrtl.module @bar(in %io_cpu_flush: !firrtl.uint<1>){
  }
  firrtl.module @unprocessedAnnotations(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>,
                            in %cond: !firrtl.uint<1>, in %value: !firrtl.uint<2>) {
    // expected-warning @+1 {{unprocessed annotation:'firrtl.transforms.RemainingAnnotation1'}}
    %1 = firrtl.wire {annotations = [{class = "firrtl.transforms.RemainingAnnotation1"}]} : !firrtl.uint<1>
    // expected-warning @+1 {{unprocessed annotation:'firrtl.transforms.RemainingAnnotation2'}}
    %2 = firrtl.node %1 {annotations = [{class = "firrtl.transforms.RemainingAnnotation2"}]} : !firrtl.uint<1>
    // expected-warning @+1 {{unprocessed annotation:'firrtl.transforms.RemainingAnnotation3'}}
    %3 = firrtl.reg %clock {annotations = [{class = "firrtl.transforms.RemainingAnnotation3"}]} : !firrtl.clock, !firrtl.uint<1>
    // expected-warning @+1 {{unprocessed annotation:'firrtl.transforms.RemainingAnnotation4'}}
    %4 = firrtl.regreset %clock, %reset, %1 {annotations = [{class = "firrtl.transforms.RemainingAnnotation4"}]} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    // expected-warning @+1 {{unprocessed annotation:'firrtl.transforms.RemainingAnnotation5'}}
    %_M_read, %_M_rw, %_M_write = firrtl.mem Undefined {depth = 12 : i64, name = "_M", portNames = ["read", "rw",
    "write"], readLatency = 0 : i32, writeLatency = 1 : i32, annotations = [{class =
    "firrtl.transforms.RemainingAnnotation5"}]} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: sint<42>, wmode: uint<1>, wdata: sint<42>, wmask: uint<1>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: sint<42>, mask: uint<1>>
    // expected-warning @+1 {{unprocessed annotation:'firrtl.transforms.RemainingAnnotation6'}}
    %5 = firrtl.instance fetch {annotations = [{class = "firrtl.transforms.RemainingAnnotation6"}]} @bar(in io_cpu_flush: !firrtl.uint<1>)
    %6 = firrtl.node %1 {annotations = [{class = "firrtl.transforms.RemainingAnnotation3"}]} : !firrtl.uint<1>
  }
}

// -----

// expected-warning @+1 {{unprocessed annotation:'circuitOpAnnotation'}}
firrtl.circuit "moduleAnno" attributes {annotations = [{class = "circuitOpAnnotation"}]} {
  // expected-warning @+1 {{unprocessed annotation:'a'}}
  firrtl.module @moduleAnno(in %io_cpu_flush: !firrtl.uint<1>) attributes
    {portAnnotations = [[{class="a"}]]} {  }
  // expected-warning @+1 {{unprocessed annotation:'b'}}
  firrtl.extmodule @extModPorts(in io_cpu_flush: !firrtl.uint<1>) attributes {portAnnotations = [[{class="b"}]]}
  // expected-warning @+1 {{unprocessed annotation:'c'}}
  firrtl.extmodule @extMod(in io_cpu_flush: !firrtl.uint<1>)
    attributes { annotations = [{class = "c"}] }
  // expected-warning @+1 {{unprocessed annotation:'d'}}
  firrtl.module @foo(in %io_cpu_flush: !firrtl.uint<1>)
    attributes { annotations = [{class = "d"}] } {}
  firrtl.module @foo2(in %io_cpu_flush: !firrtl.uint<1>)
    attributes { annotations = [{class = "b"}] } {}
  firrtl.extmodule @extModPorts2(in io_cpu_flush: !firrtl.uint<1>) attributes {portAnnotations = [[{class="c"}]]}
}

// -----

// The following annotations should be ignored and not trigger a warning
// when lowering to HW.
firrtl.circuit "Foo" attributes {annotations = [
    {class = "sifive.enterprise.firrtl.MetadataDirAnnotation", dirname = "metadata"},
    {class = "sifive.enterprise.firrtl.ElaborationArtefactsDirectory", dirname = "artefacts"},
    {class = "sifive.enterprise.firrtl.TestBenchDirAnnotation", dirname = "tb"},
    {class = "sifive.enterprise.grandcentral.phases.SubCircuitsTargetDirectory", dir = "subcircuits"},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation", directory = "gct-dir", filename = "gct-dir/bindings.sv"}
  ]} {
    firrtl.module @Foo() attributes {annotations = [
        {class = "firrtl.transforms.NoDedupAnnotation"},
        {class = "sifive.enterprise.firrtl.DontObfuscateModuleAnnotation"},
        {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"},
        {class = "sifive.enterprise.firrtl.ScalaClassAnnotation"},
        {class = "firrtl.transforms.BlackBox", circt.nonlocal = @nla_1}
    ]} {}
    // Non-local annotations should not produce errors either.
    hw.hierpath private  @nla_1 [@Bar::@s1, @Foo]
    firrtl.module @Bar() {
      firrtl.instance foo sym @s1 {annotations = [{circt.nonlocal = @nla_1, class = "circt.nonlocal"}]} @Foo()
    }
}

// -----

firrtl.circuit "SymArgZero" {
  // expected-error @+1 {{zero width port "foo" is referenced by name [#hw<innerSym@symfoo>] (e.g. in an XMR) but must be removed}}
  firrtl.module @SymArgZero(in %foo :!firrtl.uint<0> sym @symfoo) {
  }
}

// -----

firrtl.circuit "DTArgZero" {
  // expected-warning @below {{zero width port "foo" has dontTouch annotation, removing anyway}}
  firrtl.module @DTArgZero(in %foo :!firrtl.uint<0> [{class = "firrtl.transforms.DontTouchAnnotation"}]) {
  }
}

// -----

firrtl.circuit "ArgWithFieldSym" {
  // expected-error @below {{cannot lower aggregate port "foo" with field sensitive symbols, HW dialect does not support per field symbols yet}}
  firrtl.module @ArgWithFieldSym(in %foo :!firrtl.vector<uint<1>,2> sym [<@x,1,public>]) {
  }
}

// -----

firrtl.circuit "ConnectDestSubfield" {
  firrtl.module @ConnectDestSubfield(in %clock: !firrtl.clock, in %value: !firrtl.uint<1>) {
    %0 = firrtl.reg %clock : !firrtl.clock, !firrtl.bundle<a: uint<1>>
    // expected-error @below {{'hw.struct_extract' op used as connect destination}}
    %1 = firrtl.subfield %0[a] : !firrtl.bundle<a: uint<1>>
    // expected-error @below {{'firrtl.strictconnect' op LowerToHW couldn't handle this operation}}
    firrtl.strictconnect %1, %value : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "ConnectDestSubindex" {
  firrtl.module @ConnectDestSubindex(in %clock: !firrtl.clock, in %value: !firrtl.uint<1>) {
    %0 = firrtl.reg %clock : !firrtl.clock, !firrtl.vector<uint<1>, 1>
    // expected-error @below {{'hw.array_get' op used as connect destination}}
    %1 = firrtl.subindex %0[0] : !firrtl.vector<uint<1>, 1>
    // expected-error @below {{'firrtl.strictconnect' op LowerToHW couldn't handle this operation}}
    firrtl.strictconnect %1, %value : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "ConnectDestSubaccess" {
  firrtl.module @ConnectDestSubaccess(in %clock: !firrtl.clock, in %index: !firrtl.uint<1>, in %value: !firrtl.uint<1>) {
    %0 = firrtl.reg %clock : !firrtl.clock, !firrtl.vector<uint<1>, 1>
    // expected-error @below {{'hw.array_get' op used as connect destination}}
    %1 = firrtl.subaccess %0[%index] : !firrtl.vector<uint<1>, 1>, !firrtl.uint<1>
    // expected-error @below {{'firrtl.strictconnect' op LowerToHW couldn't handle this operation}}
    firrtl.strictconnect %1, %value : !firrtl.uint<1>
  }
}

// RUN: circt-opt -pass-pipeline='builtin.module(firrtl-inner-symbol-dce)' %s | FileCheck %s

// CHECK-LABEL: firrtl.circuit "Simple"
firrtl.circuit "Simple" attributes {
  annotations = [
    {
      class = "circuit",
      key = #hw.innerNameRef<@Simple::@w0>,
      dict = {key = #hw.innerNameRef<@Simple::@w1>},
      array = [#hw.innerNameRef<@Simple::@w2>],
      payload = "hello"
    }
  ]} {

  // CHECK-LABEL: firrtl.module @Simple
  firrtl.module @Simple() {
    // CHECK-NEXT: @w0
    %w0 = firrtl.wire sym @w0 : !firrtl.uint<1>
    // CHECK-NEXT: @w1
    %w1 = firrtl.wire sym @w1 : !firrtl.uint<1>
    // CHECK-NEXT: @w2
    %w2 = firrtl.wire sym @w2 : !firrtl.uint<1>
    // CHECK-NEXT: @w3
    %w3 = firrtl.wire sym @w3 : !firrtl.uint<1>
    // CHECK-NEXT: %w4
    // CHECK-NOT:  @w4
    %w4 = firrtl.wire sym @w4 : !firrtl.uint<1>

    %out, %out2, %out3 = firrtl.instance child sym @child @Child(out out: !firrtl.uint<1>, out out2: !firrtl.uint<1>, out out3: !firrtl.vector<uint<1>,2>)
    %eo, %eo2, %eo3 = firrtl.instance child sym @extchild @ExtChild(out out: !firrtl.uint<1>, out out2: !firrtl.uint<1>, out out3: !firrtl.vector<uint<1>,2>)
  }

  // CHECK-LABEL: firrtl.module @Child
  // CHECK-NOT: @deadportsym
  // CHECK-SAME: @outsym2
  // CHECK-SAME: @outsym3
  firrtl.module @Child(
    out %out : !firrtl.uint<1> sym @deadportsym,
    out %out2 : !firrtl.uint<1> sym @outsym2,
    out %out3 : !firrtl.vector<uint<1>,2> sym [<@outsym3,1,public>])
  {
    // CHECK-NEXT: @w5
    %w5 = firrtl.wire sym @w5 : !firrtl.uint<1>

    %c0_ui1 = firrtl.constant 0: !firrtl.uint<1>
    firrtl.strictconnect %out, %c0_ui1 : !firrtl.uint<1>
    firrtl.strictconnect %out2, %c0_ui1 : !firrtl.uint<1>

    // CHECK: @x
    // CHECK-NOT: @y
    %wire = firrtl.wire sym [<@x,1,public>,<@y,2,public>] : !firrtl.vector<uint<1>,2>
    firrtl.strictconnect %out3, %wire : !firrtl.vector<uint<1>,2>
  }

  // CHECK-LABEL: firrtl.extmodule @ExtChild
  // CHECK-NOT: @deadportsym
  // CHECK-SAME: @outsym2
  // CHECK-SAME: @outsym3
  firrtl.extmodule @ExtChild(
    out out : !firrtl.uint<1> sym @deadportsym,
    out out2 : !firrtl.uint<1> sym @outsym2,
    out out3 : !firrtl.vector<uint<1>,2> sym [<@outsym3,1,public>])

  hw.hierpath private @wire [@Simple::@child, @Child::@w5]
  hw.hierpath private @wireField [@Simple::@child, @Child::@x]

  hw.hierpath private @port [@Simple::@child, @Child::@outsym2]
  hw.hierpath private @portField [@Simple::@child, @Child::@outsym3]

  hw.hierpath private @extPort [@Simple::@extchild, @ExtChild::@outsym2]
  hw.hierpath private @extPortField [@Simple::@extchild, @ExtChild::@outsym3]
}

sv.verbatim "{{0}}" {symbols = [#hw.innerNameRef<@Simple::@w3>]}

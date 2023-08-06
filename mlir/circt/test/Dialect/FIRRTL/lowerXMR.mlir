// RUN: circt-opt %s  --firrtl-lower-xmr -split-input-file |  FileCheck %s

// Test for same module lowering
// CHECK-LABEL: firrtl.circuit "xmr"
firrtl.circuit "xmr" {
  // CHECK : #hw.innerNameRef<@xmr::@[[wSym]]>
  // CHECK-LABEL: firrtl.module @xmr(out %o: !firrtl.uint<2>)
  firrtl.module @xmr(out %o: !firrtl.uint<2>, in %2: !firrtl.probe<uint<0>>) {
    %w = firrtl.wire : !firrtl.uint<2>
    %1 = firrtl.ref.send %w : !firrtl.uint<2>
    %x = firrtl.ref.resolve %1 : !firrtl.probe<uint<2>>
    %x2 = firrtl.ref.resolve %2 : !firrtl.probe<uint<0>>
    // CHECK-NOT: firrtl.ref.resolve
    firrtl.strictconnect %o, %x : !firrtl.uint<2>
    // CHECK:      %w = firrtl.wire : !firrtl.uint<2>
    // CHECK:      %w_probe = firrtl.node sym @[[wSym:[a-zA-Z0-9_]+]] interesting_name %w : !firrtl.uint<2>
    // CHECK-NEXT: %[[#xmr:]] = sv.xmr.ref @xmrPath : !hw.inout<i2>
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]] : !hw.inout<i2> to !firrtl.uint<2>
    // CHECK:      firrtl.strictconnect %o, %[[#cast]] : !firrtl.uint<2>
  }
}

// -----

// Test the correct xmr path is generated
// CHECK-LABEL: firrtl.circuit "Top" {
firrtl.circuit "Top" {
  // CHECK:      hw.hierpath private @[[path:[a-zA-Z0-9_]+]]
  // CHECK-SAME:   [@Top::@bar, @Bar::@barXMR, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  firrtl.module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) {
    // CHECK: firrtl.module @XmrSrcMod() {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK:  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK:  %0 = firrtl.node sym @[[xmrSym]] %c0_ui1  : !firrtl.uint<1>
    %1 = firrtl.ref.send %zero : !firrtl.uint<1>
    firrtl.ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  firrtl.module @Bar(out %_a: !firrtl.probe<uint<1>>) {
    %xmr   = firrtl.instance bar sym @barXMR @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  firrtl.instance bar sym @barXMR  @XmrSrcMod()
    firrtl.ref.define %_a, %xmr   : !firrtl.probe<uint<1>>
  }
  firrtl.module @Top() {
    %bar_a = firrtl.instance bar sym @bar  @Bar(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  firrtl.instance bar sym @bar  @Bar()
    %a = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]] : !hw.inout<i1>
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]] : !hw.inout<i1> to !firrtl.uint<1>
    // CHECK-NEXT: firrtl.strictconnect %a, %[[#cast]] : !firrtl.uint<1>
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
  }
}

// -----

// Test 0-width xmrs are handled
// CHECK-LABEL: firrtl.circuit "Top" {
firrtl.circuit "Top" {
  firrtl.module @Top(in %bar_a : !firrtl.probe<uint<0>>, in %bar_b : !firrtl.probe<vector<uint<0>,10>>) {
    %a = firrtl.wire : !firrtl.uint<0>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.probe<uint<0>>
    // CHECK:  %[[c0_ui0:.+]] = firrtl.constant 0 : !firrtl.uint<0>
    firrtl.strictconnect %a, %0 : !firrtl.uint<0>
    // CHECK:  firrtl.strictconnect %a, %[[c0_ui0]] : !firrtl.uint<0>
    %b = firrtl.wire : !firrtl.vector<uint<0>,10>
    %1 = firrtl.ref.resolve %bar_b : !firrtl.probe<vector<uint<0>,10>>
    firrtl.strictconnect %b, %1 : !firrtl.vector<uint<0>,10>
    // CHECK:	%[[c0_ui0_0:.+]] = firrtl.constant 0 : !firrtl.uint<0>
    // CHECK:  %[[v2:.+]] = firrtl.bitcast %[[c0_ui0_0]] : (!firrtl.uint<0>) -> !firrtl.vector<uint<0>, 10>
    // CHECK:  firrtl.strictconnect %b, %[[v2]] : !firrtl.vector<uint<0>, 10>
  }
}

// -----

// Test the correct xmr path to port is generated
// CHECK-LABEL: firrtl.circuit "Top" {
firrtl.circuit "Top" {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Top::@bar, @Bar::@barXMR, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  firrtl.module @XmrSrcMod(in %pa: !firrtl.uint<1>, out %_a: !firrtl.probe<uint<1>>) {
    // CHECK: firrtl.module @XmrSrcMod(in %pa: !firrtl.uint<1>) {
    // CHECK-NEXT: firrtl.node sym @[[xmrSym]]
    %1 = firrtl.ref.send %pa : !firrtl.uint<1>
    firrtl.ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  firrtl.module @Bar(out %_a: !firrtl.probe<uint<1>>) {
    %pa, %xmr   = firrtl.instance bar sym @barXMR @XmrSrcMod(in pa: !firrtl.uint<1>, out _a: !firrtl.probe<uint<1>>)
    // CHECK: %bar_pa = firrtl.instance bar sym @barXMR  @XmrSrcMod(in pa: !firrtl.uint<1>)
    firrtl.ref.define %_a, %xmr   : !firrtl.probe<uint<1>>
  }
  firrtl.module @Top() {
    %bar_a = firrtl.instance bar sym @bar  @Bar(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  firrtl.instance bar sym @bar  @Bar()
    %a = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]] : !hw.inout<i1>
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]] : !hw.inout<i1> to !firrtl.uint<1>
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.strictconnect %a, %[[#cast]]
  }
}

// -----

// Test for multiple readers and multiple instances
// CHECK-LABEL: firrtl.circuit "Top" {
firrtl.circuit "Top" {
  // CHECK-DAG: hw.hierpath private @[[path_0:[a-zA-Z0-9_]+]] [@Foo::@fooXMR, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  // CHECK-DAG: hw.hierpath private @[[path_1:[a-zA-Z0-9_]+]] [@Bar::@barXMR, @XmrSrcMod::@[[xmrSym]]]
  // CHECK-DAG: hw.hierpath private @[[path_2:[a-zA-Z0-9_]+]] [@Top::@bar, @Bar::@barXMR, @XmrSrcMod::@[[xmrSym]]]
  // CHECK-DAG: hw.hierpath private @[[path_3:[a-zA-Z0-9_]+]] [@Top::@foo, @Foo::@fooXMR, @XmrSrcMod::@[[xmrSym]]]
  // CHECK-DAG: hw.hierpath private @[[path_4:[a-zA-Z0-9_]+]] [@Top::@xmr, @XmrSrcMod::@[[xmrSym]]]
  firrtl.module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) {
    // CHECK: firrtl.module @XmrSrcMod() {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK:   %c0_ui1 = firrtl.constant 0
    // CHECK:  %0 = firrtl.node sym @[[xmrSym]] %c0_ui1  : !firrtl.uint<1>
    %1 = firrtl.ref.send %zero : !firrtl.uint<1>
    firrtl.ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  firrtl.module @Foo(out %_a: !firrtl.probe<uint<1>>) {
    %xmr   = firrtl.instance bar sym @fooXMR @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  firrtl.instance bar sym @fooXMR  @XmrSrcMod()
    firrtl.ref.define %_a, %xmr   : !firrtl.probe<uint<1>>
    %0 = firrtl.ref.resolve %xmr   : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path_0]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %a = firrtl.wire : !firrtl.uint<1>
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
    // CHECK:      firrtl.strictconnect %a, %[[#cast]]
  }
  firrtl.module @Bar(out %_a: !firrtl.probe<uint<1>>) {
    %xmr   = firrtl.instance bar sym @barXMR @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  firrtl.instance bar sym @barXMR  @XmrSrcMod()
    firrtl.ref.define %_a, %xmr   : !firrtl.probe<uint<1>>
    %0 = firrtl.ref.resolve %xmr   : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path_1]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %a = firrtl.wire : !firrtl.uint<1>
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
    // CHECK:      firrtl.strictconnect %a, %[[#cast]]
  }
  firrtl.module @Top() {
    %bar_a = firrtl.instance bar sym @bar  @Bar(out _a: !firrtl.probe<uint<1>>)
    %foo_a = firrtl.instance foo sym @foo @Foo(out _a: !firrtl.probe<uint<1>>)
    %xmr_a = firrtl.instance xmr sym @xmr @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  firrtl.instance bar sym @bar  @Bar()
    // CHECK:  firrtl.instance foo sym @foo  @Foo()
    // CHECK:  firrtl.instance xmr sym @xmr  @XmrSrcMod()
    %a = firrtl.wire : !firrtl.uint<1>
    %b = firrtl.wire : !firrtl.uint<1>
    %c = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path_2]]
    // CHECK-NEXT: %[[#cast_2:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %1 = firrtl.ref.resolve %foo_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path_3]]
    // CHECK-NEXT: %[[#cast_3:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %2 = firrtl.ref.resolve %xmr_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path_4]]
    // CHECK-NEXT: %[[#cast_4:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.strictconnect %a, %[[#cast_2]]
    firrtl.strictconnect %b, %1 : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.strictconnect %b, %[[#cast_3]]
    firrtl.strictconnect %c, %2 : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.strictconnect %c, %[[#cast_4]]
  }
}

// -----

// Check for downward reference
// CHECK-LABEL: firrtl.circuit "Top" {
firrtl.circuit "Top" {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Top::@bar, @Bar::@barXMR, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  firrtl.module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) {
    // CHECK: firrtl.module @XmrSrcMod() {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK:  %c0_ui1 = firrtl.constant 0
    // CHECK:  %0 = firrtl.node sym @[[xmrSym]] %c0_ui1  : !firrtl.uint<1>
    %1 = firrtl.ref.send %zero : !firrtl.uint<1>
    firrtl.ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  firrtl.module @Bar(out %_a: !firrtl.probe<uint<1>>) {
    %xmr   = firrtl.instance bar sym @barXMR @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  firrtl.instance bar sym @barXMR  @XmrSrcMod()
    firrtl.ref.define %_a, %xmr   : !firrtl.probe<uint<1>>
  }
  firrtl.module @Top() {
    %bar_a = firrtl.instance bar sym @bar  @Bar(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  firrtl.instance bar sym @bar  @Bar()
    %a = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.strictconnect %a, %[[#cast]]
    %c_a = firrtl.instance child @Child(in  _a: !firrtl.probe<uint<1>>)
    firrtl.ref.define %c_a, %bar_a : !firrtl.probe<uint<1>>
  }
  firrtl.module @Child(in  %_a: !firrtl.probe<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
  }

}

// -----

// Check for downward reference to port
// CHECK-LABEL: firrtl.circuit "Top" {
firrtl.circuit "Top" {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Top::@bar, @Bar::@barXMR, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  firrtl.module @XmrSrcMod(in %pa: !firrtl.uint<1>, out %_a: !firrtl.probe<uint<1>>) {
    // CHECK: firrtl.module @XmrSrcMod(in %pa: !firrtl.uint<1>) {
    // CHECK: firrtl.node sym @[[xmrSym]]
    %1 = firrtl.ref.send %pa : !firrtl.uint<1>
    firrtl.ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  firrtl.module @Bar(out %_a: !firrtl.probe<uint<1>>) {
    %pa, %xmr   = firrtl.instance bar sym @barXMR @XmrSrcMod(in pa: !firrtl.uint<1>, out _a: !firrtl.probe<uint<1>>)
    // CHECK: %bar_pa = firrtl.instance bar sym @barXMR  @XmrSrcMod(in pa: !firrtl.uint<1>)
    firrtl.ref.define %_a, %xmr   : !firrtl.probe<uint<1>>
  }
  firrtl.module @Top() {
    %bar_a = firrtl.instance bar sym @bar  @Bar(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  firrtl.instance bar sym @bar  @Bar()
    %a = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.strictconnect %a, %[[#cast]]
    %c_a = firrtl.instance child @Child(in  _a: !firrtl.probe<uint<1>>)
    firrtl.ref.define %c_a, %bar_a : !firrtl.probe<uint<1>>
  }
  firrtl.module @Child(in  %_a: !firrtl.probe<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
  }
}

// -----

// Test for multiple paths and downward reference.
// CHECK-LABEL: firrtl.circuit "Top" {
firrtl.circuit "Top" {
  // CHECK: hw.hierpath private @[[path_0:[a-zA-Z0-9_]+]] [@Top::@foo, @Foo::@fooXMR, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  // CHECK: hw.hierpath private @[[path_1:[a-zA-Z0-9_]+]] [@Top::@xmr, @XmrSrcMod::@[[xmrSym]]]
  firrtl.module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %1 = firrtl.ref.send %zero : !firrtl.uint<1>
    firrtl.ref.define %_a, %1 : !firrtl.probe<uint<1>>
    // CHECK: firrtl.node sym @[[xmrSym]]
  }
  firrtl.module @Foo(out %_a: !firrtl.probe<uint<1>>) {
    %xmr   = firrtl.instance bar sym @fooXMR @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    firrtl.ref.define %_a, %xmr   : !firrtl.probe<uint<1>>
  }
  firrtl.module @Top() {
    %foo_a = firrtl.instance foo sym @foo @Foo(out _a: !firrtl.probe<uint<1>>)
    %xmr_a = firrtl.instance xmr sym @xmr @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    %c_a, %c_b = firrtl.instance child @Child2p(in _a: !firrtl.probe<uint<1>>, in _b: !firrtl.probe<uint<1>> )
    // CHECK:  firrtl.instance child  @Child2p()
    firrtl.ref.define %c_a, %foo_a : !firrtl.probe<uint<1>>
    firrtl.ref.define %c_b, %xmr_a : !firrtl.probe<uint<1>>
  }
  firrtl.module @Child2p(in  %_a: !firrtl.probe<uint<1>>, in  %_b: !firrtl.probe<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path_0]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %1 = firrtl.ref.resolve %_b : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path_1]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
  }
}

// -----

// Test for multiple children paths
// CHECK-LABEL: firrtl.circuit "Top" {
firrtl.circuit "Top" {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Top::@xmr, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  firrtl.module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %1 = firrtl.ref.send %zero : !firrtl.uint<1>
    firrtl.ref.define %_a, %1 : !firrtl.probe<uint<1>>
    // CHECK: firrtl.node sym @[[xmrSym]]
  }
  firrtl.module @Top() {
    %xmr_a = firrtl.instance xmr sym @xmr @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    %c_a = firrtl.instance child @Child1(in _a: !firrtl.probe<uint<1>>)
    firrtl.ref.define %c_a, %xmr_a : !firrtl.probe<uint<1>>
  }
  // CHECK-LABEL: firrtl.module @Child1() {
  firrtl.module @Child1(in  %_a: !firrtl.probe<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %c_a, %c_b = firrtl.instance child @Child2(in _a: !firrtl.probe<uint<1>>, in _b: !firrtl.probe<uint<1>> )
    firrtl.ref.define %c_a, %_a : !firrtl.probe<uint<1>>
    firrtl.ref.define %c_b, %_a : !firrtl.probe<uint<1>>
    %c3 = firrtl.instance child @Child3(in _a: !firrtl.probe<uint<1>>)
    firrtl.ref.define %c3 , %_a : !firrtl.probe<uint<1>>
  }
  firrtl.module @Child2(in  %_a: !firrtl.probe<uint<1>>, in  %_b: !firrtl.probe<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %1 = firrtl.ref.resolve %_b : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
  }
  firrtl.module @Child3(in  %_a: !firrtl.probe<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %1 = firrtl.ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
  }
}

// -----

// Test for multiple children paths
// CHECK-LABEL: firrtl.circuit "Top" {
firrtl.circuit "Top" {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Top::@xmr, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  firrtl.module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %1 = firrtl.ref.send %zero : !firrtl.uint<1>
    firrtl.ref.define %_a, %1 : !firrtl.probe<uint<1>>
    // CHECK: firrtl.node sym @[[xmrSym]]
  }
  firrtl.module @Top() {
    %xmr_a = firrtl.instance xmr sym @xmr @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    %c_a = firrtl.instance child @Child1(in _a: !firrtl.probe<uint<1>>)
    firrtl.ref.define %c_a, %xmr_a : !firrtl.probe<uint<1>>
  }
  // CHECK-LABEL: firrtl.module @Child1() {
  firrtl.module @Child1(in  %_a: !firrtl.probe<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %c_a, %c_b = firrtl.instance child @Child2(in _a: !firrtl.probe<uint<1>>, in _b: !firrtl.probe<uint<1>> )
    firrtl.ref.define %c_a, %_a : !firrtl.probe<uint<1>>
    firrtl.ref.define %c_b, %_a : !firrtl.probe<uint<1>>
    %c3 = firrtl.instance child @Child3(in _a: !firrtl.probe<uint<1>>)
    firrtl.ref.define %c3 , %_a : !firrtl.probe<uint<1>>
  }
  firrtl.module @Child2(in  %_a: !firrtl.probe<uint<1>>, in  %_b: !firrtl.probe<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %1 = firrtl.ref.resolve %_b : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
  }
  firrtl.module @Child3(in  %_a: !firrtl.probe<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %1 = firrtl.ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
  }
}

// -----

// Multiply instantiated Top works, because the reference port does not flow through it.
firrtl.circuit "Top" {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Dut::@xmr, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  firrtl.module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %1 = firrtl.ref.send %zero : !firrtl.uint<1>
    firrtl.ref.define %_a, %1 : !firrtl.probe<uint<1>>
    // CHECK: firrtl.node sym @[[xmrSym]]
  }
  firrtl.module @Top() {
    firrtl.instance d1 @Dut()
  }
  firrtl.module @Top2() {
    firrtl.instance d2 @Dut()
  }
  firrtl.module @Dut() {
    %xmr_a = firrtl.instance xmr sym @xmr @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    %c_a = firrtl.instance child @Child1(in _a: !firrtl.probe<uint<1>>)
    firrtl.ref.define %c_a, %xmr_a : !firrtl.probe<uint<1>>
  }
  // CHECK-LABEL: firrtl.module @Child1() {
  firrtl.module @Child1(in  %_a: !firrtl.probe<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %c_a, %c_b = firrtl.instance child @Child2(in _a: !firrtl.probe<uint<1>>, in _b: !firrtl.probe<uint<1>> )
    firrtl.ref.define %c_a, %_a : !firrtl.probe<uint<1>>
    firrtl.ref.define %c_b, %_a : !firrtl.probe<uint<1>>
    %c3 = firrtl.instance child @Child3(in _a: !firrtl.probe<uint<1>>)
    firrtl.ref.define %c3 , %_a : !firrtl.probe<uint<1>>
  }
  firrtl.module @Child2(in  %_a: !firrtl.probe<uint<1>>, in  %_b: !firrtl.probe<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %1 = firrtl.ref.resolve %_b : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
  }
  firrtl.module @Child3(in  %_a: !firrtl.probe<uint<1>>) {
    %0 = firrtl.ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_
    %1 = firrtl.ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_
  }
}

// -----

firrtl.circuit "Top"  {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Top::@[[TOP_XMR_SYM:.+]], @DUTModule::@[[xmrSym:[a-zA-Z0-9_]+]]]
  // CHECK-LABEL: firrtl.module private @DUTModule
  // CHECK-SAME: (in %clock: !firrtl.clock, in %io_addr: !firrtl.uint<3>, in %io_dataIn: !firrtl.uint<8>, in %io_wen: !firrtl.uint<1>, out %io_dataOut: !firrtl.uint<8>)
  firrtl.module private @DUTModule(in %clock: !firrtl.clock, in %io_addr: !firrtl.uint<3>, in %io_dataIn: !firrtl.uint<8>, in %io_wen: !firrtl.uint<1>, out %io_dataOut: !firrtl.uint<8>, out %_gen_memTap: !firrtl.probe<vector<uint<8>, 8>>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %rf_memTap, %rf_read, %rf_write = firrtl.mem  Undefined  {depth = 8 : i64, name = "rf", portNames = ["memTap", "read", "write"], prefix = "foo_", readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.probe<vector<uint<8>, 8>>, !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>, !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    // CHECK:  %rf_read, %rf_write = firrtl.mem sym @[[xmrSym]] Undefined  {depth = 8 : i64, name = "rf", portNames = ["read", "write"], prefix = "foo_", readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>, !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %0 = firrtl.subfield %rf_read[addr] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>
    %1 = firrtl.subfield %rf_read[en] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>
    %2 = firrtl.subfield %rf_read[clk] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>
    %3 = firrtl.subfield %rf_read[data] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>
    %4 = firrtl.subfield %rf_write[addr] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %5 = firrtl.subfield %rf_write[en] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %6 = firrtl.subfield %rf_write[clk] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %7 = firrtl.subfield %rf_write[data] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %8 = firrtl.subfield %rf_write[mask] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    firrtl.strictconnect %0, %io_addr : !firrtl.uint<3>
    firrtl.strictconnect %1, %c1_ui1 : !firrtl.uint<1>
    firrtl.strictconnect %2, %clock : !firrtl.clock
    firrtl.strictconnect %io_dataOut, %3 : !firrtl.uint<8>
    firrtl.strictconnect %4, %io_addr : !firrtl.uint<3>
    firrtl.strictconnect %5, %io_wen : !firrtl.uint<1>
    firrtl.strictconnect %6, %clock : !firrtl.clock
    firrtl.strictconnect %8, %c1_ui1 : !firrtl.uint<1>
    firrtl.strictconnect %7, %io_dataIn : !firrtl.uint<8>
    firrtl.ref.define %_gen_memTap, %rf_memTap : !firrtl.probe<vector<uint<8>, 8>>
  }
  // CHECK: firrtl.module @Top
  firrtl.module @Top(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %io_addr: !firrtl.uint<3>, in %io_dataIn: !firrtl.uint<8>, in %io_wen: !firrtl.uint<1>, out %io_dataOut: !firrtl.uint<8>) {
    // CHECK: firrtl.instance dut sym @[[TOP_XMR_SYM]] @DUTModule
    %dut_clock, %dut_io_addr, %dut_io_dataIn, %dut_io_wen, %dut_io_dataOut, %dut__gen_memTap = firrtl.instance dut  @DUTModule(in clock: !firrtl.clock, in io_addr: !firrtl.uint<3>, in io_dataIn: !firrtl.uint<8>, in io_wen: !firrtl.uint<1>, out io_dataOut: !firrtl.uint<8>, out _gen_memTap: !firrtl.probe<vector<uint<8>, 8>>)
    %0 = firrtl.ref.resolve %dut__gen_memTap : !firrtl.probe<vector<uint<8>, 8>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]] ".Memory"
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    firrtl.strictconnect %dut_clock, %clock : !firrtl.clock
    %memTap_0 = firrtl.wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    %memTap_1 = firrtl.wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    %memTap_2 = firrtl.wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    %memTap_3 = firrtl.wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    %memTap_4 = firrtl.wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    %memTap_5 = firrtl.wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    %memTap_6 = firrtl.wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    %memTap_7 = firrtl.wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    firrtl.strictconnect %io_dataOut, %dut_io_dataOut : !firrtl.uint<8>
    firrtl.strictconnect %dut_io_wen, %io_wen : !firrtl.uint<1>
    firrtl.strictconnect %dut_io_dataIn, %io_dataIn : !firrtl.uint<8>
    firrtl.strictconnect %dut_io_addr, %io_addr : !firrtl.uint<3>
    %1 = firrtl.subindex %0[0] : !firrtl.vector<uint<8>, 8>
    // CHECK: %[[#cast_0:]] = firrtl.subindex %[[#cast]][0]
    firrtl.strictconnect %memTap_0, %1 : !firrtl.uint<8>
    // CHECK:  firrtl.strictconnect %memTap_0, %[[#cast_0]] : !firrtl.uint<8>
    %2 = firrtl.subindex %0[1] : !firrtl.vector<uint<8>, 8>
    // CHECK: %[[#cast_1:]] = firrtl.subindex %[[#cast]][1]
    firrtl.strictconnect %memTap_1, %2 : !firrtl.uint<8>
    // CHECK:  firrtl.strictconnect %memTap_1, %[[#cast_1]] : !firrtl.uint<8>
    %3 = firrtl.subindex %0[2] : !firrtl.vector<uint<8>, 8>
    // CHECK: %[[#cast_2:]] = firrtl.subindex %[[#cast]][2]
    firrtl.strictconnect %memTap_2, %3 : !firrtl.uint<8>
    // CHECK:  firrtl.strictconnect %memTap_2, %[[#cast_2]] : !firrtl.uint<8>
    %4 = firrtl.subindex %0[3] : !firrtl.vector<uint<8>, 8>
    // CHECK: %[[#cast_3:]] = firrtl.subindex %[[#cast]][3]
    firrtl.strictconnect %memTap_3, %4 : !firrtl.uint<8>
    // CHECK:  firrtl.strictconnect %memTap_3, %[[#cast_3]] : !firrtl.uint<8>
    %5 = firrtl.subindex %0[4] : !firrtl.vector<uint<8>, 8>
    // CHECK: %[[#cast_4:]] = firrtl.subindex %[[#cast]][4]
    firrtl.strictconnect %memTap_4, %5 : !firrtl.uint<8>
    // CHECK:  firrtl.strictconnect %memTap_4, %[[#cast_4]] : !firrtl.uint<8>
    %6 = firrtl.subindex %0[5] : !firrtl.vector<uint<8>, 8>
    // CHECK: %[[#cast_5:]] = firrtl.subindex %[[#cast]][5]
    firrtl.strictconnect %memTap_5, %6 : !firrtl.uint<8>
    // CHECK:  firrtl.strictconnect %memTap_5, %[[#cast_5]] : !firrtl.uint<8>
    %7 = firrtl.subindex %0[6] : !firrtl.vector<uint<8>, 8>
    // CHECK: %[[#cast_6:]] = firrtl.subindex %[[#cast]][6]
    firrtl.strictconnect %memTap_6, %7 : !firrtl.uint<8>
    // CHECK:  firrtl.strictconnect %memTap_6, %[[#cast_6]] : !firrtl.uint<8>
    %8 = firrtl.subindex %0[7] : !firrtl.vector<uint<8>, 8>
    // CHECK: %[[#cast_7:]] = firrtl.subindex %[[#cast]][7]
    firrtl.strictconnect %memTap_7, %8 : !firrtl.uint<8>
    // CHECK:  firrtl.strictconnect %memTap_7, %[[#cast_7]] : !firrtl.uint<8>
    }
}

// -----

firrtl.circuit "Top"  {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Top::@[[TOP_XMR_SYM:.+]], @DUTModule::@[[xmrSym:[a-zA-Z0-9_]+]]]
  // CHECK-LABEL:  firrtl.module private @DUTModule
  // CHECK-SAME: in %io_wen: !firrtl.uint<1>, out %io_dataOut: !firrtl.uint<8>)
  firrtl.module private @DUTModule(in %clock: !firrtl.clock, in %io_addr: !firrtl.uint<3>, in %io_dataIn: !firrtl.uint<8>, in %io_wen: !firrtl.uint<1>, out %io_dataOut: !firrtl.uint<8>, out %_gen_memTap_0: !firrtl.probe<uint<8>>, out %_gen_memTap_1: !firrtl.probe<uint<8>>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %rf_memTap, %rf_read, %rf_write = firrtl.mem  Undefined  {depth = 2 : i64, name = "rf", portNames = ["memTap", "read", "write"], prefix = "foo_", readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.probe<vector<uint<8>, 2>>, !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<8>>, !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    // CHECK:  %rf_read, %rf_write = firrtl.mem sym @[[xmrSym]] Undefined  {depth = 2 : i64, name = "rf", portNames = ["read", "write"], prefix = "foo_", readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<8>>, !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %9 = firrtl.ref.sub %rf_memTap[0] : !firrtl.probe<vector<uint<8>, 2>>
    firrtl.ref.define %_gen_memTap_0, %9 : !firrtl.probe<uint<8>>
    %10 = firrtl.ref.sub %rf_memTap[1] : !firrtl.probe<vector<uint<8>, 2>>
    firrtl.ref.define %_gen_memTap_1, %10 : !firrtl.probe<uint<8>>
  }
  firrtl.module @Top(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %io_addr: !firrtl.uint<3>, in %io_dataIn: !firrtl.uint<8>, in %io_wen: !firrtl.uint<1>, out %io_dataOut: !firrtl.uint<8>) {
    // CHECK: firrtl.instance dut sym @[[TOP_XMR_SYM]] @DUTModule
    %dut_clock, %dut_io_addr, %dut_io_dataIn, %dut_io_wen, %dut_io_dataOut, %dut__gen_memTap_0, %dut__gen_memTap_1 = firrtl.instance dut  @DUTModule(in clock: !firrtl.clock, in io_addr: !firrtl.uint<3>, in io_dataIn: !firrtl.uint<8>, in io_wen: !firrtl.uint<1>, out io_dataOut: !firrtl.uint<8>, out _gen_memTap_0: !firrtl.probe<uint<8>>, out _gen_memTap_1: !firrtl.probe<uint<8>>)
    %0 = firrtl.ref.resolve %dut__gen_memTap_0 : !firrtl.probe<uint<8>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]] ".Memory[0]"
    // CHECK-NEXT: %[[#cast_0:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %1 = firrtl.ref.resolve %dut__gen_memTap_1 : !firrtl.probe<uint<8>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]] ".Memory[1]"
    // CHECK-NEXT: %[[#cast_1:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    firrtl.strictconnect %dut_clock, %clock : !firrtl.clock
    %memTap_0 = firrtl.wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    %memTap_1 = firrtl.wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    firrtl.strictconnect %memTap_0, %0 : !firrtl.uint<8>
    // CHECK:      firrtl.strictconnect %memTap_0, %[[#cast_0]]
    firrtl.strictconnect %memTap_1, %1 : !firrtl.uint<8>
    // CHECK:      firrtl.strictconnect %memTap_1, %[[#cast_1]]
  }
}

// -----

// Test lowering of internal path into a module
// CHECK-LABEL: firrtl.circuit "Top" {
firrtl.circuit "Top" {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Top::@bar, @Bar::@[[xmrSym:[a-zA-Z0-9_]+]]]
  firrtl.module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) {
    // CHECK: firrtl.module @XmrSrcMod() {
    // CHECK-NEXT: }
    %z = firrtl.verbatim.expr "internal.path" : () -> !firrtl.uint<1>
    %1 = firrtl.ref.send %z : !firrtl.uint<1>
    firrtl.ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  firrtl.module @Bar(out %_a: !firrtl.probe<uint<1>>) {
    %xmr   = firrtl.instance bar sym @barXMR @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  firrtl.instance bar sym @barXMR  @XmrSrcMod()
    firrtl.ref.define %_a, %xmr   : !firrtl.probe<uint<1>>
  }
  firrtl.module @Top() {
    %bar_a = firrtl.instance bar sym @bar  @Bar(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  firrtl.instance bar sym @bar  @Bar()
    %a = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]] ".internal.path"
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.strictconnect %a, %[[#cast]]
  }
}

// -----

// Test lowering of internal path into a module
// CHECK-LABEL: firrtl.circuit "Top" {
firrtl.circuit "Top" {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Top::@bar, @Bar::@barXMR, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  firrtl.module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) {
    // CHECK: firrtl.module @XmrSrcMod() {
    // CHECK{LITERAL}:  firrtl.verbatim.expr "internal.path" : () -> !firrtl.uint<1> {symbols = [@XmrSrcMod]}
    // CHECK:  = firrtl.node sym @[[xmrSym]] %[[internal:.+]]  : !firrtl.uint<1>
    %z = firrtl.verbatim.expr "internal.path" : () -> !firrtl.uint<1> {symbols = [@XmrSrcMod]}
    %1 = firrtl.ref.send %z : !firrtl.uint<1>
    firrtl.ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  firrtl.module @Bar(out %_a: !firrtl.probe<uint<1>>) {
    %xmr   = firrtl.instance bar sym @barXMR @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  firrtl.instance bar sym @barXMR  @XmrSrcMod()
    firrtl.ref.define %_a, %xmr   : !firrtl.probe<uint<1>>
  }
  firrtl.module @Top() {
    %bar_a = firrtl.instance bar sym @bar  @Bar(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  firrtl.instance bar sym @bar  @Bar()
    %a = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    firrtl.strictconnect %a, %0 : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.strictconnect %a, %[[#cast]]
  }
}

// -----

// Test correct lowering of 0-width ports
firrtl.circuit "Top"  {
  firrtl.module @XmrSrcMod(in %pa: !firrtl.uint<0>, out %_a: !firrtl.probe<uint<0>>) {
  // CHECK-LABEL: firrtl.module @XmrSrcMod(in %pa: !firrtl.uint<0>)
    %0 = firrtl.ref.send %pa : !firrtl.uint<0>
    firrtl.ref.define %_a, %0 : !firrtl.probe<uint<0>>
  }
  firrtl.module @Bar(out %_a: !firrtl.probe<uint<0>>) {
    %bar_pa, %bar__a = firrtl.instance bar sym @barXMR  @XmrSrcMod(in pa: !firrtl.uint<0>, out _a: !firrtl.probe<uint<0>>)
    firrtl.ref.define %_a, %bar__a : !firrtl.probe<uint<0>>
  }
  firrtl.module @Top() {
    %bar__a = firrtl.instance bar sym @bar  @Bar(out _a: !firrtl.probe<uint<0>>)
    %a = firrtl.wire   : !firrtl.uint<0>
    %0 = firrtl.ref.resolve %bar__a : !firrtl.probe<uint<0>>
    firrtl.strictconnect %a, %0 : !firrtl.uint<0>
    // CHECK: %c0_ui0 = firrtl.constant 0 : !firrtl.uint<0>
    // CHECK: firrtl.strictconnect %a, %c0_ui0 : !firrtl.uint<0>
  }
}

// -----
// Test lowering of XMR to instance port (result).
// https://github.com/llvm/circt/issues/4559

// CHECK-LABEL: Issue4559
firrtl.circuit "Issue4559" {
// CHECK: hw.hierpath private @xmrPath [@Issue4559::@[[SYM:.+]]]
  firrtl.extmodule @Source(out sourceport: !firrtl.uint<1>)
  firrtl.module @Issue4559() {
    // CHECK: %[[PORT:.+]] = firrtl.instance source @Source
    // CHECK-NEXT: %[[NODE:.+]] = firrtl.node sym @[[SYM]] interesting_name %[[PORT]]
    // CHECK-NEXT: = sv.xmr.ref @xmrPath
    %port = firrtl.instance source @Source(out sourceport: !firrtl.uint<1>)
    %port_ref = firrtl.ref.send %port : !firrtl.uint<1>
    %port_val = firrtl.ref.resolve %port_ref : !firrtl.probe<uint<1>>
  }
}

// -----
// Check read-only XMR of a rwprobe.

// CHECK-LABEL: firrtl.circuit "ReadForceable"
firrtl.circuit "ReadForceable" {
  // CHECK: hw.hierpath private @xmrPath [@ReadForceable::@[[wSym:.+]]]
  // CHECK: firrtl.module @ReadForceable(out %o: !firrtl.uint<2>)
  firrtl.module @ReadForceable(out %o: !firrtl.uint<2>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<2>, !firrtl.rwprobe<uint<2>>
    %x = firrtl.ref.resolve %w_ref : !firrtl.rwprobe<uint<2>>
    // CHECK-NOT: firrtl.ref.resolve
    firrtl.strictconnect %o, %x : !firrtl.uint<2>
    // CHECK:      %w, %w_ref = firrtl.wire sym @[[wSym]] forceable : !firrtl.uint<2>, !firrtl.rwprobe<uint<2>>
    // CHECK-NEXT: %[[#xmr:]] = sv.xmr.ref @xmrPath : !hw.inout<i2>
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]] : !hw.inout<i2> to !firrtl.uint<2>
    // CHECK:      firrtl.strictconnect %o, %[[#cast]] : !firrtl.uint<2>
  }
}

// -----
// Check resolution through a ref cast.

// CHECK-LABEL: firrtl.circuit "RefCast"
firrtl.circuit "RefCast" {
  // CHECK: hw.hierpath private @xmrPath [@RefCast::@[[wSym:.+]]]
  // CHECK-LABEL: firrtl.module @RefCast(out %o: !firrtl.uint<2>)
  firrtl.module @RefCast(out %o: !firrtl.uint<2>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<2>, !firrtl.rwprobe<uint<2>>
    %w_ro = firrtl.ref.cast %w_ref : (!firrtl.rwprobe<uint<2>>) -> !firrtl.probe<uint<2>>
    %x = firrtl.ref.resolve %w_ro : !firrtl.probe<uint<2>>
    firrtl.strictconnect %o, %x : !firrtl.uint<2>
    // CHECK-NEXT: %w, %w_ref = firrtl.wire sym @[[wSym]] forceable : !firrtl.uint<2>, !firrtl.rwprobe<uint<2>>
    // CHECK-NEXT: %[[#xmr:]] = sv.xmr.ref @xmrPath : !hw.inout<i2>
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]] : !hw.inout<i2> to !firrtl.uint<2>
    // CHECK-NEXT: firrtl.strictconnect %o, %[[#cast]] : !firrtl.uint<2>
  }
}

// -----

// CHECK-LABEL: firrtl.circuit "ForceRelease"
firrtl.circuit "ForceRelease" {
  // CHECK: hw.hierpath private @[[XMRPATH:.+]] [@ForceRelease::@[[INST_SYM:.+]], @RefMe::@[[TARGET_SYM:.+]]]
  // CHECK: firrtl.module private @RefMe() {
  firrtl.module private @RefMe(out %p: !firrtl.rwprobe<uint<4>>) {
    // CHECK-NEXT: %x, %x_ref = firrtl.wire sym @[[TARGET_SYM]] forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    %x, %x_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    // CHECK-NEXT: }
    firrtl.ref.define %p, %x_ref : !firrtl.rwprobe<uint<4>>
  }
  // CHECK-LABEL: firrtl.module @ForceRelease
  firrtl.module @ForceRelease(in %c: !firrtl.uint<1>, in %clock: !firrtl.clock, in %x: !firrtl.uint<4>) {
      // CHECK-NEXT: firrtl.instance r sym @[[INST_SYM]] @RefMe()
      %r_p = firrtl.instance r @RefMe(out p: !firrtl.rwprobe<uint<4>>)
      // CHECK-NEXT: %[[REF1:.+]] = sv.xmr.ref @[[XMRPATH]] : !hw.inout<i4>
      // CHECK-NEXT: %[[CAST1:.+]] = builtin.unrealized_conversion_cast %[[REF1]] : !hw.inout<i4> to !firrtl.rwprobe<uint<4>>
   
      // CHECK-NEXT: firrtl.ref.force %clock, %c, %[[CAST1]], %x : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<4>

      firrtl.ref.force %clock, %c, %r_p, %x : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<4>
      // CHECK-NEXT: %[[REF2:.+]] = sv.xmr.ref @[[XMRPATH]] : !hw.inout<i4>
      // CHECK-NEXT: %[[CAST2:.+]] = builtin.unrealized_conversion_cast %[[REF2]] : !hw.inout<i4> to !firrtl.rwprobe<uint<4>>

      // CHECK-NEXT: firrtl.ref.force_initial %c, %[[CAST2]], %x : !firrtl.uint<1>, !firrtl.uint<4>
      firrtl.ref.force_initial %c, %r_p, %x : !firrtl.uint<1>, !firrtl.uint<4>
      // CHECK-NEXT: %[[REF3:.+]] = sv.xmr.ref @[[XMRPATH]] : !hw.inout<i4>
      // CHECK-NEXT: %[[CAST3:.+]] = builtin.unrealized_conversion_cast %[[REF3]] : !hw.inout<i4> to !firrtl.rwprobe<uint<4>>
      // CHECK-NEXT: firrtl.ref.release %clock, %c, %[[CAST3]] : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>
      firrtl.ref.release %clock, %c, %r_p : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>
      // CHECK-NEXT: %[[REF4:.+]] = sv.xmr.ref @[[XMRPATH]] : !hw.inout<i4>
      // CHECK-NEXT: %[[CAST4:.+]] = builtin.unrealized_conversion_cast %[[REF4]] : !hw.inout<i4> to !firrtl.rwprobe<uint<4>>
      // CHECK-NEXT: firrtl.ref.release_initial %c, %[[CAST4]] : !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>
      firrtl.ref.release_initial %c, %r_p : !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>
    }
  }

// -----
// Check tracking of public output refs as sv.macro.decl and sv.macro.def

// CHECK-LABEL: firrtl.circuit "Top"
firrtl.circuit "Top" {
  // CHECK: sv.macro.decl @ref_Top_Top_a
  // CHECK-NEXT{LITERAL}: sv.macro.def @ref_Top_Top_a "{{0}}"
  // CHECK-SAME:          ([@[[XMR1:.*]]]) {output_file = #hw.output_file<"ref_Top_Top.sv">}

  // CHECK-NEXT:  sv.macro.decl @ref_Top_Top_b
  // CHECK-NEXT{LITERAL}: sv.macro.def @ref_Top_Top_b "{{0}}"
  // CHECK-SAME:          ([@[[XMR2:.*]]]) {output_file = #hw.output_file<"ref_Top_Top.sv">}

  // CHECK-NEXT:  sv.macro.decl @ref_Top_Top_c
  // CHECK-NEXT{LITERAL}: sv.macro.def @ref_Top_Top_c "{{0}}.internal.path"
  // CHECK-SAME:          ([@[[XMR3:.*]]]) {output_file = #hw.output_file<"ref_Top_Top.sv">}

  // CHECK-NEXT:  sv.macro.decl @ref_Top_Top_d
  // CHECK-NEXT{LITERAL}: sv.macro.def @ref_Top_Top_d "{{0}}"
  // CHECK-SAME:          ([@[[XMR4:.+]]]) {output_file = #hw.output_file<"ref_Top_Top.sv">}

  // CHECK-NOT:   sv.macro.decl @ref_Top_Top_e
  // CHECK:  hw.hierpath private @[[XMR5:.+]] [@Foo::@[[FOO_X_SYM:.+]]]
  // CHECK:  sv.macro.decl @ref_Top_Foo_x
  // CHECK-NEXT{LITERAL}: sv.macro.def @ref_Top_Foo_x "{{0}}"
  // CHECK-SAME:          ([@[[XMR5]]]) {output_file = #hw.output_file<"ref_Top_Foo.sv">}

  // CHECK-NEXT:  sv.macro.decl @ref_Top_Foo_y
  // CHECK-NEXT:          sv.macro.def @ref_Top_Foo_y "internal.path" 
  // CHECK-NOT:           ([
  // CHECK-SAME:          {output_file = #hw.output_file<"ref_Top_Foo.sv">}

  // CHECK:        hw.hierpath private @[[XMR1]] [@Top::@[[TOP_W_SYM:.+]]]
  // CHECK:        hw.hierpath private @[[XMR2]] [@Top::@foo, @Foo::@[[FOO_X_SYM]]]
  // CHECK:        hw.hierpath private @[[XMR3]] [@Top::@foo]
  // CHECK:        hw.hierpath private @[[XMR4]] [@Top::@{{.+}}]
  
  // CHECK-LABEL: firrtl.module @Top()
  firrtl.module @Top(out %a: !firrtl.probe<uint<1>>, 
                     out %b: !firrtl.probe<uint<1>>, 
                     out %c: !firrtl.probe<uint<1>>, 
                     out %d: !firrtl.probe<uint<1>>,
                     in %e: !firrtl.probe<uint<1>>) {
    %w = firrtl.wire sym @w : !firrtl.uint<1>
    // CHECK: firrtl.node sym @[[TOP_W_SYM]] interesting_name %w
    %0 = firrtl.ref.send %w : !firrtl.uint<1>
    firrtl.ref.define %a, %0 : !firrtl.probe<uint<1>>
    
    %x, %y = firrtl.instance foo sym @foo @Foo(out x: !firrtl.probe<uint<1>>, out y: !firrtl.probe<uint<1>>)
    firrtl.ref.define %b, %x : !firrtl.probe<uint<1>>
    firrtl.ref.define %c, %y : !firrtl.probe<uint<1>>
    
    %constant = firrtl.constant 0 : !firrtl.uint<1>
    %1 = firrtl.ref.send %constant : !firrtl.uint<1>
    firrtl.ref.define %d, %1 : !firrtl.probe<uint<1>>
  }

  // CHECK-LABEL: firrtl.module @Foo()
  firrtl.module @Foo(out %x: !firrtl.probe<uint<1>>, out %y: !firrtl.probe<uint<1>>) {
    %w = firrtl.wire sym @x : !firrtl.uint<1>
    // CHECK: firrtl.node sym @[[FOO_X_SYM]] interesting_name %w
    %0 = firrtl.ref.send %w : !firrtl.uint<1>
    firrtl.ref.define %x, %0 : !firrtl.probe<uint<1>>

    %z = firrtl.verbatim.expr "internal.path" : () -> !firrtl.uint<1>
    %1 = firrtl.ref.send %z : !firrtl.uint<1>
    firrtl.ref.define %y, %1 : !firrtl.probe<uint<1>>
  }
}

// -----
// Check resolving XMR's to internalPaths

// CHECK-LABEL: firrtl.circuit "InternalPaths"
firrtl.circuit "InternalPaths" {
  firrtl.extmodule private @RefExtMore(in in: !firrtl.uint<1>,
                                       out r: !firrtl.probe<uint<1>>,
                                       out data: !firrtl.uint<3>,
                                       out r2: !firrtl.probe<vector<bundle<a: uint<3>>, 3>>) attributes {convention = #firrtl<convention scalarized>, internalPaths = ["path.to.internal.signal", "in"]}
  // CHECK: hw.hierpath private @xmrPath [@InternalPaths::@[[EXT_SYM:.+]]] 
  // CHECK: module public @InternalPaths(
  firrtl.module public @InternalPaths(in %in: !firrtl.uint<1>) {
    // CHECK: firrtl.instance ext sym @[[EXT_SYM]] @RefExtMore
    %ext_in, %ext_r, %ext_data, %ext_r2 =
      firrtl.instance ext @RefExtMore(in in: !firrtl.uint<1>,
                                      out r: !firrtl.probe<uint<1>>,
                                      out data: !firrtl.uint<3>,
                                      out r2: !firrtl.probe<vector<bundle<a: uint<3>>, 3>>)
   firrtl.strictconnect %ext_in, %in : !firrtl.uint<1>

   // CHECK: %[[XMR_R:.+]] = sv.xmr.ref @xmrPath ".path.to.internal.signal" : !hw.inout<i1>
   // CHECK: %[[XMR_R_CAST:.+]] = builtin.unrealized_conversion_cast %[[XMR_R]] : !hw.inout<i1> to !firrtl.uint<1>
   // CHECK: %node_r = firrtl.node %[[XMR_R_CAST]]
   %read_r  = firrtl.ref.resolve %ext_r : !firrtl.probe<uint<1>>
   %node_r = firrtl.node %read_r : !firrtl.uint<1>
   // CHECK: %[[XMR_R2:.+]] = sv.xmr.ref @xmrPath ".in" : !hw.inout<array<3xstruct<a: i3>>>
   // CHECK: %[[XMR_R2_CAST:.+]] = builtin.unrealized_conversion_cast %[[XMR_R2]] : !hw.inout<array<3xstruct<a: i3>>> to !firrtl.vector<bundle<a: uint<3>>, 3>
   // CHECK: %node_r2 = firrtl.node %[[XMR_R2_CAST]]
   %read_r2  = firrtl.ref.resolve %ext_r2 : !firrtl.probe<vector<bundle<a: uint<3>>, 3>>
   %node_r2 = firrtl.node %read_r2 : !firrtl.vector<bundle<a: uint<3>>, 3>
  }
}

// -----
// Check resolving XMR's to use macro ABI.

// CHECK-LABEL: firrtl.circuit "RefABI"
firrtl.circuit "RefABI" {
  firrtl.extmodule private @RefExtMore(in in: !firrtl.uint<1>,
                                       out r: !firrtl.probe<uint<1>>,
                                       out data: !firrtl.uint<3>,
                                       out r2: !firrtl.probe<vector<bundle<a: uint<3>>, 3>>) attributes {convention = #firrtl<convention scalarized>}
  // CHECK:  hw.hierpath private @xmrPath [@RefABI::@[[XMR_SYM:.+]]] 
  // CHECK: module public @RefABI(
  firrtl.module public @RefABI(in %in: !firrtl.uint<1>) {
    %ext_in, %ext_r, %ext_data, %ext_r2 =
      // CHECK: firrtl.instance ext sym @[[XMR_SYM]] @RefExtMore
      firrtl.instance ext @RefExtMore(in in: !firrtl.uint<1>,
                                      out r: !firrtl.probe<uint<1>>,
                                      out data: !firrtl.uint<3>,
                                      out r2: !firrtl.probe<vector<bundle<a: uint<3>>, 3>>)
   firrtl.strictconnect %ext_in, %in : !firrtl.uint<1>

   // CHECK: %[[XMR_R:.+]] = sv.xmr.ref @xmrPath ".`ref_RefExtMore_RefExtMore_r" : !hw.inout<i1>
   // CHECK: %[[XMR_R_CAST:.+]] = builtin.unrealized_conversion_cast %[[XMR_R]] : !hw.inout<i1> to !firrtl.uint<1>
   // CHECK: %node_r = firrtl.node %[[XMR_R_CAST]]
   %read_r  = firrtl.ref.resolve %ext_r : !firrtl.probe<uint<1>>
   %node_r = firrtl.node %read_r : !firrtl.uint<1>
   // CHECK: %[[XMR_R2:.+]] = sv.xmr.ref @xmrPath ".`ref_RefExtMore_RefExtMore_r2" : !hw.inout<array<3xstruct<a: i3>>>
   // CHECK: %[[XMR_R2_CAST:.+]] = builtin.unrealized_conversion_cast %[[XMR_R2]] : !hw.inout<array<3xstruct<a: i3>>> to !firrtl.vector<bundle<a: uint<3>>, 3>
   // CHECK: %node_r2 = firrtl.node %[[XMR_R2_CAST]]
   %read_r2  = firrtl.ref.resolve %ext_r2 : !firrtl.probe<vector<bundle<a: uint<3>>, 3>>
   %node_r2 = firrtl.node %read_r2 : !firrtl.vector<bundle<a: uint<3>>, 3>
  }
}

// -----
// Check handling of basic ref.sub.

// CHECK-LABEL: circuit "BasicRefSub"
firrtl.circuit "BasicRefSub" {
  // CHECK:  hw.hierpath private @[[XMRPATH:.+]] [@BasicRefSub::@[[C_SYM:[^,]+]], @Child::@[[REF_SYM:[^,]+]]]
  // CHECK-LABEL: firrtl.module private @Child
  // CHECK-SAME: in %in: !firrtl.bundle<a: uint<1>, b: uint<2>>)
  firrtl.module private @Child(in %in : !firrtl.bundle<a: uint<1>, b: uint<2>>, out %out : !firrtl.probe<uint<2>>) {
    // CHECK-NEXT: firrtl.node sym @[[REF_SYM]] interesting_name %in
    %ref = firrtl.ref.send %in : !firrtl.bundle<a: uint<1>, b: uint<2>>
    %sub = firrtl.ref.sub %ref[1] : !firrtl.probe<bundle<a: uint<1>, b: uint<2>>>
    firrtl.ref.define %out, %sub : !firrtl.probe<uint<2>>
  }
  // CHECK-LABEL: module @BasicRefSub(
  firrtl.module @BasicRefSub(in %in : !firrtl.bundle<a: uint<1>, b: uint<2>>, out %out : !firrtl.uint<2>) {
    // CHECK: firrtl.instance c sym @[[C_SYM]]
    %c_in, %c_out = firrtl.instance c @Child(in in : !firrtl.bundle<a: uint<1>, b: uint<2>>, out out : !firrtl.probe<uint<2>>)
    firrtl.strictconnect %c_in, %in : !firrtl.bundle<a: uint<1>, b: uint<2>>
    // CHECK: sv.xmr.ref @[[XMRPATH]] ".b"
    %res = firrtl.ref.resolve %c_out : !firrtl.probe<uint<2>>
    firrtl.strictconnect %out, %res : !firrtl.uint<2>
  }
}

// -----
// Check rwprobe, forceable, ABI

// CHECK-LABEL: circuit "RWProbe_field"
firrtl.circuit "RWProbe_field" {
  // CHECK: hw.hierpath private @[[XMRPATH:.+]] [@RWProbe_field::@[[SYM:[^,]+]]]
  // CHECK-NEXT: sv.macro.decl @ref_RWProbe_field_RWProbe_field_rw
  // e.g., "n[0]"
  // CHECK-NEXT{LITERAL}: sv.macro.def @ref_RWProbe_field_RWProbe_field_rw "{{0}}[0]"
  // CHECK-SAME: ([@[[XMRPATH]]])
  // CHECK-NEXT: sv.macro.decl @ref_RWProbe_field_RWProbe_field_rw_narrow
  // e.g., "n[0].a"
  // CHECK-NEXT{LITERAL}: sv.macro.def @ref_RWProbe_field_RWProbe_field_rw_narrow "{{0}}[0].a"
  // CHECK-SAME: ([@[[XMRPATH]]])
  firrtl.module @RWProbe_field(in %x: !firrtl.vector<bundle<a: uint<1>>, 2>, out %rw: !firrtl.rwprobe<bundle<a: uint<1>>>, out %rw_narrow : !firrtl.rwprobe<uint<1>>) {
    %n, %n_ref = firrtl.node %x forceable : !firrtl.vector<bundle<a: uint<1>>, 2>
    %0 = firrtl.ref.sub %n_ref[0] : !firrtl.rwprobe<vector<bundle<a: uint<1>>, 2>>
    firrtl.ref.define %rw, %0 : !firrtl.rwprobe<bundle<a: uint<1>>>
    %1 = firrtl.ref.sub %0[0] : !firrtl.rwprobe<bundle<a: uint<1>>>
    firrtl.ref.define %rw_narrow, %1 : !firrtl.rwprobe<uint<1>>
  }
}

// -----
// Check ref.sub handling through layers, combining with import/export ABI (if aggs preserved?).

// CHECK-LABEL: circuit "RefSubLayers"
firrtl.circuit "RefSubLayers" {
  // CHECK: hw.hierpath private @[[XMRPATH:.+]] [@RefSubLayers::@[[TOP_SYM:[^,]+]], @Mid::@[[MID_SYM:[^,]+]], @Leaf::@[[LEAF_SYM:.+]]]
  // CHECK-NEXT: sv.macro.decl @ref_RefSubLayers_RefSubLayers_rw
  // CHECK-NEXT{LITERAL}: sv.macro.def @ref_RefSubLayers_RefSubLayers_rw "{{0}}.`ref_ExtRef_ExtRef_out.b[1].a"
  // CHECK-SAME: ([@[[XMRPATH]]])
   firrtl.extmodule @ExtRef(out out: !firrtl.probe<bundle<a: uint<1>, b: vector<bundle<a: uint<2>, b: uint<1>>, 2>>>)
  firrtl.module @RefSubLayers(out %rw : !firrtl.probe<uint<2>>) {
    %ref = firrtl.instance m @Mid(out rw: !firrtl.probe<bundle<a: uint<2>, b: uint<1>>>)
    %sub = firrtl.ref.sub %ref[0] : !firrtl.probe<bundle<a: uint<2>, b: uint<1>>>
    firrtl.ref.define %rw, %sub : !firrtl.probe<uint<2>>
  }
  firrtl.module private @Mid(out %rw : !firrtl.probe<bundle<a: uint<2>, b: uint<1>>>) {
    %ref = firrtl.instance l @Leaf(out rw: !firrtl.probe<vector<bundle<a: uint<2>, b: uint<1>>, 2>>)
    %sub = firrtl.ref.sub %ref[1] : !firrtl.probe<vector<bundle<a: uint<2>, b: uint<1>>, 2>>
    firrtl.ref.define %rw, %sub : !firrtl.probe<bundle<a: uint<2>, b: uint<1>>>
  }

  firrtl.module private @Leaf(out %rw : !firrtl.probe<vector<bundle<a: uint<2>, b: uint<1>>, 2>>) {
    %ref = firrtl.instance ext @ExtRef(out out: !firrtl.probe<bundle<a: uint<1>, b: vector<bundle<a: uint<2>, b: uint<1>>, 2>>>)
    %sub = firrtl.ref.sub %ref[1] : !firrtl.probe<bundle<a: uint<1>, b: vector<bundle<a: uint<2>, b: uint<1>>, 2>>>
    firrtl.ref.define %rw, %sub : !firrtl.probe<vector<bundle<a: uint<2>, b: uint<1>>, 2>>
  }
}

// -----
// Check dropping force/etc. ops that target zero-width references.
// Ensure no symbol added, so can be dropped in LowerToHW.

// CHECK-LABEL: circuit "DropForceOp"
firrtl.circuit "DropForceOp" {
  firrtl.module @DropForceOp() {
    // CHECK: firrtl.wire
    // CHECK-NOT: sym
    // CHECK-NEXT: }
    %c0_ui0 = firrtl.constant 0 : !firrtl.uint<0>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %x, %x_ref = firrtl.wire forceable : !firrtl.uint<0>, !firrtl.rwprobe<uint<0>>
    firrtl.ref.force_initial %c1_ui1, %x_ref, %c0_ui0 : !firrtl.uint<1>, !firrtl.uint<0>
  }
}

// -----
// Check dropping zero-width and interaction with ABI/across layers.

// CHECK-LABEL: circuit "RefSubZeroWidth"
firrtl.circuit "RefSubZeroWidth" {
   // CHECK-NOT: probe<
   firrtl.extmodule @ExtRef(out out: !firrtl.probe<bundle<a: uint<1>, b: vector<bundle<a: uint<0>, b: uint<1>>, 2>>>)
  firrtl.module @RefSubZeroWidth(out %rw : !firrtl.probe<uint<0>>) {
    %ref = firrtl.instance m @Mid(out rw: !firrtl.probe<bundle<a: uint<0>, b: uint<1>>>)
    %sub = firrtl.ref.sub %ref[0] : !firrtl.probe<bundle<a: uint<0>, b: uint<1>>>
    firrtl.ref.define %rw, %sub : !firrtl.probe<uint<0>>
  }
  firrtl.module private @Mid(out %rw : !firrtl.probe<bundle<a: uint<0>, b: uint<1>>>) {
    %ref = firrtl.instance l @Leaf(out rw: !firrtl.probe<vector<bundle<a: uint<0>, b: uint<1>>, 2>>)
    %sub = firrtl.ref.sub %ref[1] : !firrtl.probe<vector<bundle<a: uint<0>, b: uint<1>>, 2>>
    firrtl.ref.define %rw, %sub : !firrtl.probe<bundle<a: uint<0>, b: uint<1>>>
  }

  firrtl.module private @Leaf(out %rw : !firrtl.probe<vector<bundle<a: uint<0>, b: uint<1>>, 2>>) {
    %ref = firrtl.instance ext @ExtRef(out out: !firrtl.probe<bundle<a: uint<1>, b: vector<bundle<a: uint<0>, b: uint<1>>, 2>>>)
    %sub = firrtl.ref.sub %ref[1] : !firrtl.probe<bundle<a: uint<1>, b: vector<bundle<a: uint<0>, b: uint<1>>, 2>>>
    firrtl.ref.define %rw, %sub : !firrtl.probe<vector<bundle<a: uint<0>, b: uint<1>>, 2>>
  }
}

// -----
// Check resolving through rwprobe ops, particularly when pointing to specific field.
// CHECK-LABEL: circuit "RWProbePort"

firrtl.circuit "RWProbePort" {
  // CHECK:  hw.hierpath private @[[XMRPATH:.+]] [@RWProbePort::@target]
  // CHECK{LITERAL}: sv.macro.def @ref_RWProbePort_RWProbePort_p "{{0}}"
  // CHECK-SAME: ([@[[XMRPATH]]])
  // CHECK: module @RWProbePort(
  // CHECK-NOT: firrtl.ref.rwprobe
  // CHECK-NEXT: }
  firrtl.module @RWProbePort(in %in: !firrtl.vector<uint<1>, 2> sym [<@target,2,public>], out %p: !firrtl.rwprobe<uint<1>>) {
    %0 = firrtl.ref.rwprobe <@RWProbePort::@target> : !firrtl.uint<1>
    firrtl.ref.define %p, %0 : !firrtl.rwprobe<uint<1>>
  }
}

// -----
// Test resolving through output ports through points that aren't handled by unification.
// (ref.sub).

// CHECK-LABEL: circuit "RefSubOutputPort"
firrtl.circuit "RefSubOutputPort" {
  // CHECK: hw.hierpath private @[[XMRPATH:.+]] [@RefSubOutputPort::@[[CHILD_SYM:.+]], @Child::@[[WIRE_SYM:.+]]]
  // CHECK: sv.macro.def @ref_RefSubOutputPort_RefSubOutputPort_outVec
  // CHECK-SAME{LITERAL}: "{{0}}.x"
  // CHECK-SAME: ([@[[XMRPATH]]]) {output_file = #hw.output_file<"ref_RefSubOutputPort_RefSubOutputPort.sv">}
  // CHECK: sv.macro.def @ref_RefSubOutputPort_RefSubOutputPort_outElem
  // CHECK-SAME{LITERAL}: "{{0}}.x[1]"
  // CHECK-SAME: ([@[[XMRPATH]]]) {output_file = #hw.output_file<"ref_RefSubOutputPort_RefSubOutputPort.sv">}
  // CHECK: sv.macro.def @ref_RefSubOutputPort_RefSubOutputPort_outElemDirect
  // CHECK-SAME{LITERAL}: "{{0}}.x[1]"
  // CHECK-SAME: ([@[[XMRPATH]]]) {output_file = #hw.output_file<"ref_RefSubOutputPort_RefSubOutputPort.sv">}

  // CHECK: module private @Child
  // CHECK-NEXT: firrtl.wire sym @[[WIRE_SYM]] forceable
  firrtl.module private @Child(out %bore_1: !firrtl.rwprobe<bundle<x: vector<uint<1>, 2>>>) {
    %b, %b_ref = firrtl.wire forceable : !firrtl.bundle<x: vector<uint<1>, 2>>, !firrtl.rwprobe<bundle<x: vector<uint<1>, 2>>>
    firrtl.ref.define %bore_1, %b_ref : !firrtl.rwprobe<bundle<x: vector<uint<1>, 2>>>
  }
  // CHECK: module @RefSubOutputPort
  firrtl.module @RefSubOutputPort(out %outRWBundleProbe: !firrtl.rwprobe<bundle<x: vector<uint<1>, 2>>>,
                                  out %outVec: !firrtl.rwprobe<vector<uint<1>, 2>>,
                                  out %outElem: !firrtl.rwprobe<uint<1>>,
                                  out %outElemDirect: !firrtl.rwprobe<uint<1>>) attributes {convention = #firrtl<convention scalarized>} {
    %0 = firrtl.ref.sub %outVec[1] : !firrtl.rwprobe<vector<uint<1>, 2>>
    %1 = firrtl.ref.sub %outRWBundleProbe[0] : !firrtl.rwprobe<bundle<x: vector<uint<1>, 2>>>
    %2 = firrtl.ref.sub %1[1] : !firrtl.rwprobe<vector<uint<1>, 2>>
    // CHECK-NEXT: instance child sym @[[CHILD_SYM]] @Child
    // CHECK-NEXT: }
    %child_bore_1 = firrtl.instance child @Child(out bore_1: !firrtl.rwprobe<bundle<x: vector<uint<1>, 2>>>)
    firrtl.ref.define %outRWBundleProbe, %child_bore_1 : !firrtl.rwprobe<bundle<x: vector<uint<1>, 2>>>
    firrtl.ref.define %outElemDirect, %2 : !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %outVec, %1 : !firrtl.rwprobe<vector<uint<1>, 2>>
    firrtl.ref.define %outElem, %0 : !firrtl.rwprobe<uint<1>>
  }
}

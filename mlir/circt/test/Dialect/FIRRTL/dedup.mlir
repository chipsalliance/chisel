// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-dedup))' %s | FileCheck %s

// CHECK-LABEL: firrtl.circuit "Empty"
firrtl.circuit "Empty" {
  // CHECK: firrtl.module @Empty0
  firrtl.module @Empty0(in %i0: !firrtl.uint<1>) { }
  // CHECK-NOT: firrtl.module @Empty1
  firrtl.module @Empty1(in %i1: !firrtl.uint<1>) { }
  // CHECK-NOT: firrtl.module @Empty2
  firrtl.module @Empty2(in %i2: !firrtl.uint<1>) { }
  firrtl.module @Empty() {
    // CHECK: %e0_i0 = firrtl.instance e0  @Empty0
    // CHECK: %e1_i0 = firrtl.instance e1  @Empty0
    // CHECK: %e2_i0 = firrtl.instance e2  @Empty0
    %e0_i0 = firrtl.instance e0 @Empty0(in i0: !firrtl.uint<1>)
    %e1_i1 = firrtl.instance e1 @Empty1(in i1: !firrtl.uint<1>)
    %e2_i2 = firrtl.instance e2 @Empty2(in i2: !firrtl.uint<1>)
  }
}


// CHECK-LABEL: firrtl.circuit "Simple"
firrtl.circuit "Simple" {
  // CHECK: firrtl.module @Simple0
  firrtl.module @Simple0() {
    %a = firrtl.wire: !firrtl.bundle<a: uint<1>>
  }
  // CHECK-NOT: firrtl.module @Simple1
  firrtl.module @Simple1() {
    %b = firrtl.wire: !firrtl.bundle<b: uint<1>>
  }
  firrtl.module @Simple() {
    // CHECK: firrtl.instance simple0 @Simple0
    // CHECK: firrtl.instance simple1 @Simple0
    firrtl.instance simple0 @Simple0()
    firrtl.instance simple1 @Simple1()
  }
}

// CHECK-LABEL: firrtl.circuit "PrimOps"
firrtl.circuit "PrimOps" {
  // CHECK: firrtl.module @PrimOps0
  firrtl.module @PrimOps0(in %a: !firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>) {
    %a_a = firrtl.subfield %a[a] : !firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>
    %a_b = firrtl.subfield %a[b] : !firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>
    %a_c = firrtl.subfield %a[c] : !firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>
    %0 = firrtl.xor %a_a, %a_b: (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
    firrtl.connect %a_c, %a_b: !firrtl.uint<2>, !firrtl.uint<2>
  }
  // CHECK-NOT: firrtl.module @PrimOps1
  firrtl.module @PrimOps1(in %b: !firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>) {
    %b_a = firrtl.subfield %b[a] : !firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>
    %b_b = firrtl.subfield %b[b] : !firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>
    %b_c = firrtl.subfield %b[c] : !firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>
    %0 = firrtl.xor %b_a, %b_b: (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
    firrtl.connect %b_c, %b_b: !firrtl.uint<2>, !firrtl.uint<2>
  }
  firrtl.module @PrimOps() {
    // CHECK: firrtl.instance primops0 @PrimOps0
    // CHECK: firrtl.instance primops1 @PrimOps0
    %primops0 = firrtl.instance primops0 @PrimOps0(in a: !firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>)
    %primops1 = firrtl.instance primops1 @PrimOps1(in b: !firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>)
  }
}

// Check that when operations are recursively merged.
// CHECK-LABEL: firrtl.circuit "WhenOps"
firrtl.circuit "WhenOps" {
  // CHECK: firrtl.module @WhenOps0
  firrtl.module @WhenOps0(in %p : !firrtl.uint<1>) {
    // CHECK: firrtl.when %p : !firrtl.uint<1> {
    // CHECK:  %w = firrtl.wire : !firrtl.uint<8>
    // CHECK: }
    firrtl.when %p : !firrtl.uint<1> {
      %w = firrtl.wire : !firrtl.uint<8>
    }
  }
  // CHECK-NOT: firrtl.module @PrimOps1
  firrtl.module @WhenOps1(in %p : !firrtl.uint<1>) {
    firrtl.when %p : !firrtl.uint<1> {
      %w = firrtl.wire : !firrtl.uint<8>
    }
  }
  firrtl.module @WhenOps() {
    // CHECK: firrtl.instance whenops0 @WhenOps0
    // CHECK: firrtl.instance whenops1 @WhenOps0
    %whenops0 = firrtl.instance whenops0 @WhenOps0(in p : !firrtl.uint<1>)
    %whenops1 = firrtl.instance whenops1 @WhenOps1(in p : !firrtl.uint<1>)
  }
}

// CHECK-LABEL: firrtl.circuit "Annotations"
firrtl.circuit "Annotations" {
  // CHECK: hw.hierpath private [[NLA0:@nla.*]] [@Annotations::@annotations1, @Annotations0]
  // CHECK: hw.hierpath private @annos_nla0 [@Annotations::@annotations0, @Annotations0::@c]
  hw.hierpath private @annos_nla0 [@Annotations::@annotations0, @Annotations0::@c]
  // CHECK: hw.hierpath private @annos_nla1 [@Annotations::@annotations1, @Annotations0::@c]
  hw.hierpath private @annos_nla1 [@Annotations::@annotations1, @Annotations1::@j]
  // CHECK: hw.hierpath private @annos_nla2 [@Annotations::@annotations0, @Annotations0]
  hw.hierpath private @annos_nla2 [@Annotations::@annotations0, @Annotations0]

  // CHECK: firrtl.module @Annotations0() attributes {annotations = [{circt.nonlocal = [[NLA0]], class = "one"}]}
  firrtl.module @Annotations0() {
    // Annotation from other module becomes non-local.
    // CHECK: %a = firrtl.wire {annotations = [{circt.nonlocal = [[NLA0]], class = "one"}]}
    %a = firrtl.wire : !firrtl.uint<1>

    // Annotation from this module becomes non-local.
    // CHECK: %b = firrtl.wire {annotations = [{circt.nonlocal = @annos_nla2, class = "one"}]}
    %b = firrtl.wire {annotations = [{class = "one"}]} : !firrtl.uint<1>

    // Two non-local annotations are unchanged, as they have enough context in the NLA already.
    // CHECK: %c = firrtl.wire sym @c  {annotations = [{circt.nonlocal = @annos_nla0, class = "NonLocal"}, {circt.nonlocal = @annos_nla1, class = "NonLocal"}]}
    %c = firrtl.wire sym @c {annotations = [{circt.nonlocal = @annos_nla0, class = "NonLocal"}]} : !firrtl.uint<1>

    // Same test as above but with the hiearchical path targeting the module.
    // CHECK: %d = firrtl.wire {annotations = [{circt.nonlocal = @annos_nla2, class = "NonLocal"}, {circt.nonlocal = @annos_nla2, class = "NonLocal"}]}
    %d = firrtl.wire {annotations = [{circt.nonlocal = @annos_nla2, class = "NonLocal"}]} : !firrtl.uint<1>

    // Same annotation on both ops should become non-local.
    // CHECK: %e = firrtl.wire {annotations = [{circt.nonlocal = @annos_nla2, class = "both"}, {circt.nonlocal = [[NLA0]], class = "both"}]}
    %e = firrtl.wire {annotations = [{class = "both"}]} : !firrtl.uint<1>

    // Dont touch on both ops should become local.
    // CHECK: %f = firrtl.wire  {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]}
    %f = firrtl.wire {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<1>

    // Subannotations should be handled correctly.
    // CHECK: %g = firrtl.wire {annotations = [{circt.fieldID = 1 : i32, circt.nonlocal = @annos_nla2, class = "subanno"}]}
    %g = firrtl.wire {annotations = [{circt.fieldID = 1 : i32, class = "subanno"}]} : !firrtl.bundle<a: uint<1>>
  }
  // CHECK-NOT: firrtl.module @Annotations1
  firrtl.module @Annotations1() attributes {annotations = [{class = "one"}]} {
    %h = firrtl.wire {annotations = [{class = "one"}]} : !firrtl.uint<1>
    %i = firrtl.wire : !firrtl.uint<1>
    %j = firrtl.wire sym @j {annotations = [{circt.nonlocal = @annos_nla1, class = "NonLocal"}]} : !firrtl.uint<1>
    %k = firrtl.wire {annotations = [{circt.nonlocal = @annos_nla2, class = "NonLocal"}]} : !firrtl.uint<1>
    %l = firrtl.wire {annotations = [{class = "both"}]} : !firrtl.uint<1>
    %m = firrtl.wire {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<1>
    %n = firrtl.wire : !firrtl.bundle<a: uint<1>>
  }
  firrtl.module @Annotations() {
    // CHECK: firrtl.instance annotations0 sym @annotations0  @Annotations0()
    // CHECK: firrtl.instance annotations1 sym @annotations1  @Annotations0()
    firrtl.instance annotations0 sym @annotations0 @Annotations0()
    firrtl.instance annotations1 sym @annotations1 @Annotations1()
  }
}

// Special handling of DontTouch.
// CHECK-LABEL: firrtl.circuit "DontTouch"
firrtl.circuit "DontTouch" {
hw.hierpath private @nla0 [@DontTouch::@bar, @Bar::@auto]
hw.hierpath private @nla1 [@DontTouch::@baz, @Baz::@auto]
firrtl.module @DontTouch() {
  // CHECK: %bar_auto = firrtl.instance bar sym @bar @Bar(out auto: !firrtl.bundle<a: uint<1>, b: uint<1>>)
  // CHECK: %baz_auto = firrtl.instance baz sym @baz @Bar(out auto: !firrtl.bundle<a: uint<1>, b: uint<1>>)
  %bar_auto = firrtl.instance bar sym @bar @Bar(out auto: !firrtl.bundle<a: uint<1>, b: uint<1>>)
  %baz_auto = firrtl.instance baz sym @baz @Baz(out auto: !firrtl.bundle<a: uint<1>, b: uint<1>>)
}
// CHECK:      firrtl.module private @Bar(
// CHECK-SAME:   out %auto: !firrtl.bundle<a: uint<1>, b: uint<1>> sym @auto
// CHECK-SAME:   [{circt.fieldID = 1 : i32, class = "firrtl.transforms.DontTouchAnnotation"},
// CHECK-SAME:    {circt.fieldID = 2 : i32, class = "firrtl.transforms.DontTouchAnnotation"}]) {
firrtl.module private @Bar(out %auto: !firrtl.bundle<a: uint<1>, b: uint<1>> sym @auto
  [{circt.nonlocal = @nla0, circt.fieldID = 1 : i32, class = "firrtl.transforms.DontTouchAnnotation"},
  {circt.fieldID = 2 : i32, class = "firrtl.transforms.DontTouchAnnotation"}]) { }
// CHECK-NOT: firrtl.module private @Baz
firrtl.module private @Baz(out %auto: !firrtl.bundle<a: uint<1>, b: uint<1>> sym @auto
  [{circt.fieldID = 1 : i32, class = "firrtl.transforms.DontTouchAnnotation"},
  {circt.nonlocal = @nla1, circt.fieldID = 2 : i32, class = "firrtl.transforms.DontTouchAnnotation"}]) { }
}


// Check that module and memory port annotations are merged correctly.
// CHECK-LABEL: firrtl.circuit "PortAnnotations"
firrtl.circuit "PortAnnotations" {
  // CHECK: hw.hierpath private [[NLA1:@nla.*]] [@PortAnnotations::@[[PORTANNOS_1:.+]], @PortAnnotations0]
  // CHECK: hw.hierpath private [[NLA0:@nla.*]] [@PortAnnotations::@[[PORTANNOS_0:.+]], @PortAnnotations0]
  // CHECK: firrtl.module @PortAnnotations0(in %a: !firrtl.uint<1> [
  // CHECK-SAME: {circt.nonlocal = [[NLA0]], class = "port0"},
  // CHECK-SAME: {circt.nonlocal = [[NLA1]], class = "port1"}]) {
  firrtl.module @PortAnnotations0(in %a : !firrtl.uint<1> [{class = "port0"}]) {
    // CHECK: %bar_r = firrtl.mem
    // CHECK-SAME: portAnnotations =
    // CHECK-SAME:  {circt.nonlocal = [[NLA0]], class = "mem0"},
    // CHECK-SAME:  {circt.nonlocal = [[NLA1]], class = "mem1"}
    %bar_r = firrtl.mem Undefined  {depth = 16 : i64, name = "bar", portAnnotations = [[{class = "mem0"}]], portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
  }
  // CHECK-NOT: firrtl.module @PortAnnotations1
  firrtl.module @PortAnnotations1(in %b : !firrtl.uint<1> [{class = "port1"}])  {
    %bar_r = firrtl.mem Undefined  {depth = 16 : i64, name = "bar", portAnnotations = [[{class = "mem1"}]], portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
  }
  // CHECK: firrtl.module @PortAnnotations
  firrtl.module @PortAnnotations() {
    // CHECK: firrtl.instance portannos0 sym @[[PORTANNOS_0]] @PortAnnotations
    %portannos0_in = firrtl.instance portannos0 @PortAnnotations0(in a: !firrtl.uint<1>)
    // CHECK: firrtl.instance portannos1 sym @[[PORTANNOS_1]] @PortAnnotations
    %portannos1_in = firrtl.instance portannos1 @PortAnnotations1(in b: !firrtl.uint<1>)
  }
}

// Non-local annotations should have their path updated and bread crumbs should
// not be turned into non-local annotations. Note that this should not create
// totally new NLAs for the annotations, it should just update the existing
// ones.
// CHECK-LABEL: firrtl.circuit "Breadcrumb"
firrtl.circuit "Breadcrumb" {
  // CHECK:  @breadcrumb_nla0 [@Breadcrumb::@breadcrumb0, @Breadcrumb0::@crumb0, @Crumb::@in]
  hw.hierpath private @breadcrumb_nla0 [@Breadcrumb::@breadcrumb0, @Breadcrumb0::@crumb0, @Crumb::@in]
  // CHECK:  @breadcrumb_nla1 [@Breadcrumb::@breadcrumb1, @Breadcrumb0::@crumb0, @Crumb::@in]
  hw.hierpath private @breadcrumb_nla1 [@Breadcrumb::@breadcrumb1, @Breadcrumb1::@crumb1, @Crumb::@in]
  // CHECK:  @breadcrumb_nla2 [@Breadcrumb::@breadcrumb0, @Breadcrumb0::@crumb0, @Crumb::@w]
  hw.hierpath private @breadcrumb_nla2 [@Breadcrumb::@breadcrumb0, @Breadcrumb0::@crumb0, @Crumb::@w]
  // CHECK:  @breadcrumb_nla3 [@Breadcrumb::@breadcrumb1, @Breadcrumb0::@crumb0, @Crumb::@w]
  hw.hierpath private @breadcrumb_nla3 [@Breadcrumb::@breadcrumb1, @Breadcrumb1::@crumb1, @Crumb::@w]
  firrtl.module @Crumb(in %in: !firrtl.uint<1> sym @in [
      {circt.nonlocal = @breadcrumb_nla0, class = "port0"},
      {circt.nonlocal = @breadcrumb_nla1, class = "port1"}]) {
    %w = firrtl.wire sym @w {annotations = [
      {circt.nonlocal = @breadcrumb_nla2, class = "wire0"},
      {circt.nonlocal = @breadcrumb_nla3, class = "wire1"}]}: !firrtl.uint<1>
  }
  // CHECK: firrtl.module @Breadcrumb0()
  firrtl.module @Breadcrumb0() {
    // CHECK: %crumb0_in = firrtl.instance crumb0 sym @crumb0
    %crumb_in = firrtl.instance crumb0 sym @crumb0 @Crumb(in in : !firrtl.uint<1>)
  }
  // CHECK-NOT: firrtl.module @Breadcrumb1()
  firrtl.module @Breadcrumb1() {
    %crumb_in = firrtl.instance crumb1 sym @crumb1 @Crumb(in in : !firrtl.uint<1>)
  }
  // CHECK: firrtl.module @Breadcrumb()
  firrtl.module @Breadcrumb() {
    firrtl.instance breadcrumb0 sym @breadcrumb0 @Breadcrumb0()
    firrtl.instance breadcrumb1 sym @breadcrumb1 @Breadcrumb1()
  }
}

// Non-local annotations should be updated with additional context if the module
// at the root of the NLA is deduplicated.  The original NLA should be deleted,
// and the annotation should be cloned for each parent of the root module.
// CHECK-LABEL: firrtl.circuit "Context"
firrtl.circuit "Context" {
  // CHECK: hw.hierpath private [[NLA3:@nla.*]] [@Context::@[[CONTEXT1:.+]], @Context0::@c0, @ContextLeaf::@w]
  // CHECK: hw.hierpath private [[NLA1:@nla.*]] [@Context::@[[CONTEXT1]], @Context0::@c0, @ContextLeaf::@in]
  // CHECK: hw.hierpath private [[NLA2:@nla.*]] [@Context::@[[CONTEXT0:.+]], @Context0::@c0, @ContextLeaf::@w]
  // CHECK: hw.hierpath private [[NLA0:@nla.*]] [@Context::@[[CONTEXT0]], @Context0::@c0, @ContextLeaf::@in]
  // CHECK-NOT: @context_nla0
  // CHECK-NOT: @context_nla1
  // CHECK-NOT: @context_nla2
  // CHECK-NOT: @context_nla3
  hw.hierpath private @context_nla0 [@Context0::@c0, @ContextLeaf::@in]
  hw.hierpath private @context_nla1 [@Context0::@c0, @ContextLeaf::@w]
  hw.hierpath private @context_nla2 [@Context1::@c1, @ContextLeaf::@in]
  hw.hierpath private @context_nla3 [@Context1::@c1, @ContextLeaf::@w]

  // CHECK: firrtl.module @ContextLeaf(in %in: !firrtl.uint<1> sym @in [
  // CHECK-SAME: {circt.nonlocal = [[NLA0]], class = "port0"},
  // CHECK-SAME: {circt.nonlocal = [[NLA1]], class = "port1"}]
  firrtl.module @ContextLeaf(in %in : !firrtl.uint<1> sym @in [
      {circt.nonlocal = @context_nla0, class = "port0"},
      {circt.nonlocal = @context_nla2, class = "port1"}
    ]) {

    // CHECK: %w = firrtl.wire sym @w  {annotations = [
    // CHECK-SAME: {circt.nonlocal = [[NLA2]], class = "fake0"}
    // CHECK-SAME: {circt.nonlocal = [[NLA3]], class = "fake1"}
    %w = firrtl.wire sym @w {annotations = [
      {circt.nonlocal = @context_nla1, class = "fake0"},
      {circt.nonlocal = @context_nla3, class = "fake1"}]}: !firrtl.uint<3>
  }
  firrtl.module @Context0() {
    // CHECK: %leaf_in = firrtl.instance leaf sym @c0
    %leaf_in = firrtl.instance leaf sym @c0 @ContextLeaf(in in : !firrtl.uint<1>)
  }
  // CHECK-NOT: firrtl.module @Context1()
  firrtl.module @Context1() {
    %leaf_in = firrtl.instance leaf sym @c1 @ContextLeaf(in in : !firrtl.uint<1>)
  }
  firrtl.module @Context() {
    // CHECK: firrtl.instance context0 sym @[[CONTEXT0]]
    firrtl.instance context0 @Context0()
    // CHECK: firrtl.instance context1 sym @[[CONTEXT1]]
    firrtl.instance context1 @Context1()
  }
}


// When an annotation is already non-local, and is copied over to another
// module, and in further dedups force us to add more context to the
// hierarchical path, the target of the annotation should be updated to use the
// new NLA.
// CHECK-LABEL: firrtl.circuit "Context"
firrtl.circuit "Context" {

  // CHECK-NOT: hw.hierpath private @nla0
  hw.hierpath private @nla0 [@Context0::@leaf0, @ContextLeaf0::@w0]
  // CHECK-NOT: hw.hierpath private @nla1
  hw.hierpath private @nla1 [@Context1::@leaf1, @ContextLeaf1::@w1]

  // CHECK: hw.hierpath private [[NLA0:@.+]] [@Context::@[[CONTEXT1:.+]], @Context0::@leaf0, @ContextLeaf0::@w0]
  // CHECK: hw.hierpath private [[NLA1:@.+]] [@Context::@[[CONTEXT0:.+]], @Context0::@leaf0, @ContextLeaf0::@w0]

  // CHECK: firrtl.module @ContextLeaf0()
  firrtl.module @ContextLeaf0() {
    // CHECK: %w0 = firrtl.wire sym @w0  {annotations = [
    // CHECK-SAME: {circt.nonlocal = [[NLA1]], class = "fake0"}
    // CHECK-SAME: {circt.nonlocal = [[NLA0]], class = "fake1"}]}
    %w0 = firrtl.wire sym @w0 {annotations = [
      {circt.nonlocal = @nla0, class = "fake0"}]}: !firrtl.uint<3>
  }

  firrtl.module @ContextLeaf1() {
    %w1 = firrtl.wire sym @w1 {annotations = [
      {circt.nonlocal = @nla1, class = "fake1"}]}: !firrtl.uint<3>
  }

  firrtl.module @Context0() {
    firrtl.instance leaf0 sym @leaf0 @ContextLeaf0()
  }

  firrtl.module @Context1() {
    firrtl.instance leaf1 sym @leaf1 @ContextLeaf1()
  }

  firrtl.module @Context() {
    // CHECK: firrtl.instance context0 sym @[[CONTEXT0]]
    firrtl.instance context0 @Context0()
    // CHECK: firrtl.instance context1 sym @[[CONTEXT1]]
    firrtl.instance context1 @Context1()
  }
}


// This is a larger version of the above test using 3 modules.
// CHECK-LABEL: firrtl.circuit "DuplicateNLAs"
firrtl.circuit "DuplicateNLAs" {
  // CHECK-NOT: hw.hierpath private @annos_nla_1 [@Mid_1::@core, @Core_1]
  // CHECK-NOT: hw.hierpath private @annos_nla_2 [@Mid_2::@core, @Core_2]
  // CHECK-NOT: hw.hierpath private @annos_nla_3 [@Mid_3::@core, @Core_3]
  hw.hierpath private @annos_nla_1 [@Mid_1::@core, @Core_1]
  hw.hierpath private @annos_nla_2 [@Mid_2::@core, @Core_2]
  hw.hierpath private @annos_nla_3 [@Mid_3::@core, @Core_3]

  // CHECK: hw.hierpath private [[NLA0:@.+]] [@DuplicateNLAs::@core_3, @Mid_1::@core, @Core_1]
  // CHECK: hw.hierpath private [[NLA1:@.+]] [@DuplicateNLAs::@core_2, @Mid_1::@core, @Core_1]
  // CHECK: hw.hierpath private [[NLA2:@.+]] [@DuplicateNLAs::@core_1, @Mid_1::@core, @Core_1]

  firrtl.module @DuplicateNLAs() {
    firrtl.instance core_1 sym @core_1 @Mid_1()
    firrtl.instance core_2 sym @core_2 @Mid_2()
    firrtl.instance core_3 sym @core_3 @Mid_3()
  }

  firrtl.module private @Mid_1() {
    firrtl.instance core sym @core @Core_1()
  }

  firrtl.module private @Mid_2() {
    firrtl.instance core sym @core @Core_2()
  }

  firrtl.module private @Mid_3() {
    firrtl.instance core sym @core @Core_3()
  }

  // CHECK: firrtl.module private @Core_1() attributes {annotations = [
  // CHECK-SAME: {circt.nonlocal = [[NLA2]], class = "SomeAnno1"}
  // CHECK-SAME: {circt.nonlocal = [[NLA1]], class = "SomeAnno2"}
  // CHECK-SAME: {circt.nonlocal = [[NLA0]], class = "SomeAnno3"}
  firrtl.module private @Core_1() attributes {
    annotations = [
      {circt.nonlocal = @annos_nla_1, class = "SomeAnno1"}
    ]
  } { }

  firrtl.module private @Core_2() attributes {
    annotations = [
      {circt.nonlocal = @annos_nla_2, class = "SomeAnno2"}
    ]
  } { }

  firrtl.module private @Core_3() attributes {
    annotations = [
      {circt.nonlocal = @annos_nla_3, class = "SomeAnno3"}
    ]
  } { }
}

// External modules should dedup and fixup any NLAs.
// CHECK: firrtl.circuit "ExtModuleTest"
firrtl.circuit "ExtModuleTest" {
  // CHECK: hw.hierpath private @ext_nla [@ExtModuleTest::@e1, @ExtMod0]
  hw.hierpath private @ext_nla [@ExtModuleTest::@e1, @ExtMod1]
  // CHECK: firrtl.extmodule @ExtMod0() attributes {annotations = [{circt.nonlocal = @ext_nla}], defname = "a"}
  firrtl.extmodule @ExtMod0() attributes {defname = "a"}
  // CHECK-NOT: firrtl.extmodule @ExtMod1()
  firrtl.extmodule @ExtMod1() attributes {annotations = [{circt.nonlocal = @ext_nla}], defname = "a"}
  firrtl.module @ExtModuleTest() {
    // CHECK: firrtl.instance e0  @ExtMod0()
    firrtl.instance e0 @ExtMod0()
    // CHECK: firrtl.instance e1 sym @e1 @ExtMod0()
    firrtl.instance e1 sym @e1 @ExtMod1()
  }
}

// External modules with NLAs on ports should be properly rewritten.
// https://github.com/llvm/circt/issues/2713
// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "Foo"  {
  // CHECK: hw.hierpath private @nla_1 [@Foo::@b, @A::@a]
  hw.hierpath private @nla_1 [@Foo::@b, @B::@b]
  // CHECK: firrtl.extmodule @A(out a: !firrtl.clock sym @a [{circt.nonlocal = @nla_1}])
  firrtl.extmodule @A(out a: !firrtl.clock)
  firrtl.extmodule @B(out b: !firrtl.clock sym @b [{circt.nonlocal = @nla_1}])
  firrtl.module @Foo() {
    %b0_out = firrtl.instance a @A(out a: !firrtl.clock)
    // CHECK: firrtl.instance b sym @b  @A(out a: !firrtl.clock)
    %b1_out = firrtl.instance b sym @b @B(out b: !firrtl.clock)
  }
}

// Extmodules should properly hash port types and not dedup when they differ.
// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "Foo"  {
  // CHECK: firrtl.extmodule @Bar
  firrtl.extmodule @Bar(
    in clock: !firrtl.clock, out io: !firrtl.bundle<a: clock>)
  // CHECK: firrtl.extmodule @Baz
  firrtl.extmodule @Baz(
    in clock: !firrtl.clock, out io: !firrtl.bundle<a flip: uint<1>, b flip: uint<16>, c: uint<1>>)
  firrtl.module @Foo() {
    %bar_clock, %bar_io = firrtl.instance bar @Bar(
      in clock: !firrtl.clock, out io: !firrtl.bundle<a: clock>)
    %baz_clock, %baz_io = firrtl.instance baz @Baz(
      in clock: !firrtl.clock, out io: !firrtl.bundle<a flip: uint<1>, b flip: uint<16>, c: uint<1>>)
  }
}

// As we dedup modules, the chain on NLAs should continuously grow.
// CHECK-LABEL: firrtl.circuit "Chain"
firrtl.circuit "Chain" {
  // CHECK: hw.hierpath private [[NLA1:@nla.*]] [@Chain::@[[CHAINB1:.+]], @ChainB0::@[[CHAINA0:.+]], @ChainA0::@[[EXTCHAIN0:.+]], @ExtChain0]
  // CHECK: hw.hierpath private [[NLA0:@nla.*]] [@Chain::@[[CHAINB0:.+]], @ChainB0::@[[CHAINA0:.+]], @ChainA0::@[[EXTCHAIN0:.+]], @ExtChain0]
  // CHECK: firrtl.module @ChainB0()
  firrtl.module @ChainB0() {
    // CHECK: firrtl.instance chainA0 sym @[[CHAINA0]]
    firrtl.instance chainA0 @ChainA0()
  }
  // CHECK: firrtl.extmodule @ExtChain0() attributes {annotations = [
  // CHECK-SAME:  {circt.nonlocal = [[NLA0]], class = "0"},
  // CHECK-SAME:  {circt.nonlocal = [[NLA1]], class = "1"}], defname = "ExtChain"}
  firrtl.extmodule @ExtChain0() attributes {annotations = [{class = "0"}], defname = "ExtChain"}
  // CHECK-NOT: firrtl.extmodule @ExtChain1()
  firrtl.extmodule @ExtChain1() attributes {annotations = [{class = "1"}], defname = "ExtChain"}
  // CHECK: firrtl.module @ChainA0()
  firrtl.module @ChainA0()  {
    // CHECK: instance extchain0 sym @[[EXTCHAIN0]]
    firrtl.instance extchain0 @ExtChain0()
  }
  // CHECK-NOT: firrtl.module @ChainB1()
  firrtl.module @ChainB1() {
    firrtl.instance chainA1 @ChainA1()
  }
  // CHECK-NOT: firrtl.module @ChainA1()
  firrtl.module @ChainA1()  {
    firrtl.instance extchain1 @ExtChain1()
  }
  firrtl.module @Chain() {
    // CHECK: firrtl.instance chainB0 sym @[[CHAINB0]] @ChainB0()
    firrtl.instance chainB0 @ChainB0()
    // CHECK: firrtl.instance chainB1 sym @[[CHAINB1]] @ChainB0()
    firrtl.instance chainB1 @ChainB1()
  }
}


// Check that we fixup subfields and connects, when an
// instance op starts returning a different bundle type.
// CHECK-LABEL: firrtl.circuit "Bundle"
firrtl.circuit "Bundle" {
  // CHECK: firrtl.module @Bundle0
  firrtl.module @Bundle0(out %a: !firrtl.bundle<b: bundle<c flip: uint<1>, d: uint<1>>>) { }
  // CHECK-NOT: firrtl.module @Bundle1
  firrtl.module @Bundle1(out %e: !firrtl.bundle<f: bundle<g flip: uint<1>, h: uint<1>>>) { }
  firrtl.module @Bundle() {
    // CHECK: firrtl.instance bundle0  @Bundle0
    %a = firrtl.instance bundle0 @Bundle0(out a: !firrtl.bundle<b: bundle<c flip: uint<1>, d: uint<1>>>)

    // CHECK: firrtl.instance bundle1  @Bundle0
    // CHECK: %a = firrtl.wire : !firrtl.bundle<f: bundle<g flip: uint<1>, h: uint<1>>>
    // CHECK: [[A_F:%.+]] = firrtl.subfield %a[f]
    // CHECK: [[A_B:%.+]] = firrtl.subfield %bundle1_a[b]
    // CHECK: [[A_F_G:%.+]] = firrtl.subfield %0[g]
    // CHECK: [[A_B_C:%.+]] = firrtl.subfield %1[c]
    // CHECK: firrtl.strictconnect [[A_B_C]], [[A_F_G]]
    // CHECK: [[A_F_H:%.+]] = firrtl.subfield [[A_F]][h]
    // CHECK: [[A_B_D:%.+]] = firrtl.subfield [[A_B]][d]
    // CHECK: firrtl.strictconnect [[A_F_H]], [[A_B_D]]
    %e = firrtl.instance bundle1 @Bundle1(out e: !firrtl.bundle<f: bundle<g flip: uint<1>, h: uint<1>>>)

    // CHECK: [[B:%.+]] = firrtl.subfield %bundle0_a[b]
    %b = firrtl.subfield %a[b] : !firrtl.bundle<b: bundle<c flip: uint<1>, d: uint<1>>>

    // CHECK: [[F:%.+]] = firrtl.subfield %a[f]
    %f = firrtl.subfield %e[f] : !firrtl.bundle<f: bundle<g flip: uint<1>, h: uint<1>>>

    // Check that we properly fixup connects when the field names change.
    %w0 = firrtl.wire : !firrtl.bundle<g flip: uint<1>, h: uint<1>>

    // CHECK: firrtl.connect %w0, [[F]]
    firrtl.connect %w0, %f : !firrtl.bundle<g flip: uint<1>, h: uint<1>>, !firrtl.bundle<g flip: uint<1>, h: uint<1>>
  }
}

// CHECK-LABEL: firrtl.circuit "MuxBundle"
firrtl.circuit "MuxBundle" {
  firrtl.module @Bar0(out %o: !firrtl.bundle<a: uint<1>>) {
    %invalid = firrtl.invalidvalue : !firrtl.bundle<a: uint<1>>
    firrtl.strictconnect %o, %invalid : !firrtl.bundle<a: uint<1>>
  }
  firrtl.module @Bar1(out %o: !firrtl.bundle<b: uint<1>>) {
    %invalid = firrtl.invalidvalue : !firrtl.bundle<b: uint<1>>
    firrtl.strictconnect %o, %invalid : !firrtl.bundle<b: uint<1>>
  }
  firrtl.module @MuxBundle(in %p: !firrtl.uint<1>, in %l: !firrtl.bundle<b: uint<1>>, out %o: !firrtl.bundle<b: uint<1>>) attributes {convention = #firrtl<convention scalarized>} {
    // CHECK: %bar0_o = firrtl.instance bar0 @Bar0(out o: !firrtl.bundle<a: uint<1>>)
    %bar0_o = firrtl.instance bar0 @Bar0(out o: !firrtl.bundle<a: uint<1>>)

    // CHECK: %bar1_o = firrtl.instance bar1 @Bar0(out o: !firrtl.bundle<a: uint<1>>)
    // CHECK: [[WIRE:%.+]] = firrtl.wire {name = "o"} : !firrtl.bundle<b: uint<1>>
    // CHECK: [[WIRE_B:%.+]] = firrtl.subfield [[WIRE]][b]
    // CHECK: [[PORT_A:%.+]] = firrtl.subfield %bar1_o[a]
    // CHECK: firrtl.strictconnect [[WIRE_B]], [[PORT_A]]
    %bar1_o = firrtl.instance bar1 @Bar1(out o: !firrtl.bundle<b: uint<1>>)

    // CHECK: %2 = firrtl.mux(%p, [[WIRE]], %l)
    // CHECK: firrtl.strictconnect %o, %2 : !firrtl.bundle<b: uint<1>>
    %0 = firrtl.mux(%p, %bar1_o, %l) : (!firrtl.uint<1>, !firrtl.bundle<b: uint<1>>, !firrtl.bundle<b: uint<1>>) -> !firrtl.bundle<b: uint<1>>
    firrtl.strictconnect %o, %0 : !firrtl.bundle<b: uint<1>>
  }
}

// Make sure flipped fields are handled properly. This should pass flow
// verification checking.
// CHECK-LABEL: firrtl.circuit "Flip"
firrtl.circuit "Flip" {
  firrtl.module @Flip0(out %io: !firrtl.bundle<foo flip: uint<1>, fuzz: uint<1>>) {
    %0 = firrtl.subfield %io[foo] : !firrtl.bundle<foo flip: uint<1>, fuzz: uint<1>>
    %1 = firrtl.subfield %io[fuzz] : !firrtl.bundle<foo flip: uint<1>, fuzz: uint<1>>
    firrtl.connect %1, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  firrtl.module @Flip1(out %io: !firrtl.bundle<bar flip: uint<1>, buzz: uint<1>>) {
    %0 = firrtl.subfield %io[bar] : !firrtl.bundle<bar flip: uint<1>, buzz: uint<1>>
    %1 = firrtl.subfield %io[buzz] : !firrtl.bundle<bar flip: uint<1>, buzz: uint<1>>
    firrtl.connect %1, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  firrtl.module @Flip(out %io: !firrtl.bundle<foo: bundle<foo flip: uint<1>, fuzz: uint<1>>, bar: bundle<bar flip: uint<1>, buzz: uint<1>>>) {
    %0 = firrtl.subfield %io[bar] : !firrtl.bundle<foo: bundle<foo flip: uint<1>, fuzz: uint<1>>, bar: bundle<bar flip: uint<1>, buzz: uint<1>>>
    %1 = firrtl.subfield %io[foo] : !firrtl.bundle<foo: bundle<foo flip: uint<1>, fuzz: uint<1>>, bar: bundle<bar flip: uint<1>, buzz: uint<1>>>
    %foo_io = firrtl.instance foo  @Flip0(out io: !firrtl.bundle<foo flip: uint<1>, fuzz: uint<1>>)
    %bar_io = firrtl.instance bar  @Flip1(out io: !firrtl.bundle<bar flip: uint<1>, buzz: uint<1>>)
    firrtl.connect %1, %foo_io : !firrtl.bundle<foo flip: uint<1>, fuzz: uint<1>>, !firrtl.bundle<foo flip: uint<1>, fuzz: uint<1>>
    firrtl.connect %0, %bar_io : !firrtl.bundle<bar flip: uint<1>, buzz: uint<1>>, !firrtl.bundle<bar flip: uint<1>, buzz: uint<1>>
  }
}

// This is checking that the fixup phase due to changing bundle names does not
// block the deduplication of parent modules.
// CHECK-LABEL: firrtl.circuit "DelayedFixup"
firrtl.circuit "DelayedFixup"  {
  // CHECK: firrtl.extmodule @Foo
  firrtl.extmodule @Foo(out a: !firrtl.bundle<a: uint<1>>)
  // CHECK-NOT: firrtl.extmodule @Bar
  firrtl.extmodule @Bar(out b: !firrtl.bundle<b: uint<1>>)
  // CHECK: firrtl.module @Parent0
  firrtl.module @Parent0(out %a: !firrtl.bundle<a: uint<1>>, out %b: !firrtl.bundle<b: uint<1>>) {
    %foo_a = firrtl.instance foo  @Foo(out a: !firrtl.bundle<a: uint<1>>)
    firrtl.connect %a, %foo_a : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
    %bar_b = firrtl.instance bar  @Bar(out b: !firrtl.bundle<b: uint<1>>)
    firrtl.connect %b, %bar_b : !firrtl.bundle<b: uint<1>>, !firrtl.bundle<b: uint<1>>
  }
  // CHECK-NOT: firrtl.module @Parent1
  firrtl.module @Parent1(out %a: !firrtl.bundle<a: uint<1>>, out %b: !firrtl.bundle<b: uint<1>>) {
    %foo_a = firrtl.instance foo  @Foo(out a: !firrtl.bundle<a: uint<1>>)
    firrtl.connect %a, %foo_a : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
    %bar_b = firrtl.instance bar  @Bar(out b: !firrtl.bundle<b: uint<1>>)
    firrtl.connect %b, %bar_b : !firrtl.bundle<b: uint<1>>, !firrtl.bundle<b: uint<1>>
  }
  firrtl.module @DelayedFixup() {
    // CHECK: firrtl.instance parent0  @Parent0
    %parent0_a, %parent0_b = firrtl.instance parent0  @Parent0(out a: !firrtl.bundle<a: uint<1>>, out b: !firrtl.bundle<b: uint<1>>)
    // CHECK: firrtl.instance parent1  @Parent0
    %parent1_a, %parent1_b = firrtl.instance parent1  @Parent1(out a: !firrtl.bundle<a: uint<1>>, out b: !firrtl.bundle<b: uint<1>>)
  }
}

// Don't attach empty annotations onto ops without annotations.
// CHECK-LABEL: firrtl.circuit "NoEmptyAnnos"
firrtl.circuit "NoEmptyAnnos" {
  // CHECK-LABEL: @NoEmptyAnnos0()
  firrtl.module @NoEmptyAnnos0() {
    // CHECK: %w = firrtl.wire  : !firrtl.bundle<a: uint<1>>
    // CHECK: %0 = firrtl.subfield %w[a] : !firrtl.bundle<a: uint<1>>
    %w = firrtl.wire : !firrtl.bundle<a: uint<1>>
    %0 = firrtl.subfield %w[a] : !firrtl.bundle<a: uint<1>>
  }
  firrtl.module @NoEmptyAnnos1() {
    %w = firrtl.wire : !firrtl.bundle<a: uint<1>>
    %0 = firrtl.subfield %w[a] : !firrtl.bundle<a: uint<1>>
  }
  firrtl.module @NoEmptyAnnos() {
    firrtl.instance empty0 @NoEmptyAnnos0()
    firrtl.instance empty1 @NoEmptyAnnos1()
  }
}


// Don't deduplicate modules with NoDedup.
// CHECK-LABEL: firrtl.circuit "NoDedup"
firrtl.circuit "NoDedup" {
  firrtl.module @Simple0() { }
  firrtl.module @Simple1() attributes {annotations = [{class = "firrtl.transforms.NoDedupAnnotation"}]} { }
  // CHECK: firrtl.module @NoDedup
  firrtl.module @NoDedup() {
    firrtl.instance simple0 @Simple0()
    firrtl.instance simple1 @Simple1()
  }
}

// Don't deduplicate modules with input RefType ports.
// CHECK-LABEL:   firrtl.circuit "InputRefTypePorts"
// CHECK-COUNT-3: firrtl.module
firrtl.circuit "InputRefTypePorts" {
  firrtl.module @Foo(in %a: !firrtl.probe<uint<1>>) {}
  firrtl.module @Bar(in %a: !firrtl.probe<uint<1>>) {}
  firrtl.module @InputRefTypePorts() {
    %foo_a = firrtl.instance foo @Foo(in a: !firrtl.probe<uint<1>>)
    %bar_a = firrtl.instance bar @Bar(in a: !firrtl.probe<uint<1>>)
  }
}

// Check that modules marked MustDedup have been deduped.
// CHECK-LABEL: firrtl.circuit "MustDedup"
firrtl.circuit "MustDedup" attributes {annotations = [{
    // The annotation should be removed.
    // CHECK-NOT: class = "firrtl.transforms.MustDeduplicateAnnotation"
    class = "firrtl.transforms.MustDeduplicateAnnotation",
    modules = ["~MustDedup|Simple0", "~MustDedup|Simple1"]}]
   } {
  // CHECK: @Simple0
  firrtl.module @Simple0() { }
  // CHECK-NOT: @Simple1
  firrtl.module @Simple1() { }
  // CHECK: firrtl.module @MustDedup
  firrtl.module @MustDedup() {
    firrtl.instance simple0 @Simple0()
    firrtl.instance simple1 @Simple1()
  }
}

// Check that the following doesn't crash.
// https://github.com/llvm/circt/issues/3360
firrtl.circuit "Foo"  {
  firrtl.module private @X() { }
  firrtl.module private @Y() { }
  firrtl.module @Foo() {
    firrtl.instance x0 @X()
    firrtl.instance y0 @Y()
    firrtl.instance y1 @Y()
  }
}

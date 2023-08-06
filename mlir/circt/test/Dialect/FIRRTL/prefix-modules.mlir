// RUN: circt-opt --pass-pipeline="builtin.module(firrtl.circuit(firrtl-prefix-modules))" %s | FileCheck %s

// Check that the circuit is updated when the main module is updated.
// CHECK: firrtl.circuit "T_Top"
firrtl.circuit "Top" {
  // CHECK: firrtl.module @T_Top
  firrtl.module @Top()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = true
    }]} {
  }
}


// Check that the circuit is not updated if the annotation is non-inclusive.
// CHECK: firrtl.circuit "Top"
firrtl.circuit "Top" {
  // CHECK: firrtl.module @Top
  firrtl.module @Top()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = false
    }]} {
  }
}


// Check that basic module prefixing is working.
firrtl.circuit "Top" {
  // The annotation should be removed.
  // CHECK:  firrtl.module @Top() {
  firrtl.module @Top()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = false
    }]} {
    firrtl.instance test @Zebra()
  }

  // CHECK: firrtl.module @T_Zebra
  // CHECK-NOT: firrtl.module @Zebra
  firrtl.module @Zebra() { }
}


// Check that memories are renamed.
firrtl.circuit "Top" {
  firrtl.module @Top()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = true
    }]} {
    // CHECK: firrtl.mem
    // CHECK-SAME: name = "ram1"
    // CHECK-SAME: prefix = "T_"
    %ram1_r = firrtl.mem Undefined {depth = 256 : i64, name = "ram1", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<8>, en: uint<1>, clk: clock, data flip: uint<1>>
    // CHECK: firrtl.mem
    // CHECK-SAME: name = "ram2"
    // CHECK-SAME: prefix = "T_foo_"
    %ram2_r = firrtl.mem Undefined {depth = 256 : i64, name = "ram2", portNames = ["r"], prefix = "foo_", readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<8>, en: uint<1>, clk: clock, data flip: uint<1>>
  }
}

// Check that memory modules are renamed.
// CHECK-LABEL: firrtl.circuit "MemModule"
firrtl.circuit "MemModule" {
  // CHECK: firrtl.memmodule @T_MWrite_ext
  firrtl.memmodule @MWrite_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>) attributes {dataWidth = 42 : ui32, depth = 12 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
  firrtl.module @MemModule()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = false
    }]} {
    // CHECK: firrtl.instance MWrite_ext  @T_MWrite_ext
    %0:4 = firrtl.instance MWrite_ext  @MWrite_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>)
  }
}

// Check that external modules are not renamed.
// CHECK: firrtl.circuit "T_Top"
firrtl.circuit "Top" {
  // CHECK: firrtl.extmodule @ExternalModule
  firrtl.extmodule @ExternalModule()

  // CHECK: firrtl.module @T_Top
  firrtl.module @Top()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = true
    }]} {

    firrtl.instance ext @ExternalModule()
  }
}


// Check that the module is not cloned more than necessary.
firrtl.circuit "Top0" {
  firrtl.module @Top0()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = false
    }]} {
    firrtl.instance test @Zebra()
  }

  firrtl.module @Top1()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = false
    }]} {
    firrtl.instance test @Zebra()
  }

  // CHECK: firrtl.module @T_Zebra
  // CHECK-NOT: firrtl.module @T_Zebra
  // CHECK-NOT: firrtl.module @Zebra
  firrtl.module @Zebra() { }
}


// Complex nested test.
// CHECK: firrtl.circuit "T_Top"
firrtl.circuit "Top" {
  // CHECK: firrtl.module @T_Top
  firrtl.module @Top()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = true
    }]} {

    // CHECK: firrtl.instance test @T_Aardvark()
    firrtl.instance test @Aardvark()

    // CHECK: firrtl.instance test @T_Z_Zebra()
    firrtl.instance test @Zebra()
  }

  // CHECK: firrtl.module @T_Aardvark
  firrtl.module @Aardvark()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "A_",
      inclusive = false
    }]} {

    // CHECK: firrtl.instance test @T_A_Z_Zebra()
    firrtl.instance test @Zebra()
  }

  // CHECK: firrtl.module @T_Z_Zebra
  // CHECK: firrtl.module @T_A_Z_Zebra
  firrtl.module @Zebra()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "Z_",
      inclusive = true
    }]} {
  }
}


// Updates should be made to a Grand Central interface to add a "prefix" field.
// The annotatinos associated with the parent and companion should be
// unmodified.
// CHECK-LABEL: firrtl.circuit "GCTInterfacePrefix"
// CHECK-SAME:    name = "MyView", prefix = "FOO_"
firrtl.circuit "GCTInterfacePrefix"
  attributes {annotations = [{
    class = "sifive.enterprise.grandcentral.AugmentedBundleType",
    defName = "MyInterface",
    elements = [],
    id = 0 : i64,
    name = "MyView"}]}  {
  // CHECK:      firrtl.module @FOO_MyView_companion
  // CHECK-SAME:   name = "MyView"
  firrtl.module @MyView_companion()
    attributes {annotations = [{
      class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
      id = 0 : i64,
      name = "MyView",
      type = "companion"}]} {}
  // CHECK:      firrtl.module @FOO_DUT
  // CHECK-SAME:   name = "MyView"
  firrtl.module @DUT()
    attributes {annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation.parent",
       id = 0 : i64,
       name = "MyView",
       type = "parent"},
      {class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
       prefix = "FOO_",
       inclusive = true}]} {
    firrtl.instance MyView_companion  @MyView_companion()
  }
  firrtl.module @GCTInterfacePrefix() {
    firrtl.instance dut @DUT()
  }
}

// CHECK: firrtl.circuit "T_NLATop"
firrtl.circuit "NLATop" {

  hw.hierpath private @nla [@NLATop::@test, @Aardvark::@test, @Zebra]
  hw.hierpath private @nla_1 [@NLATop::@test,@Aardvark::@test_1, @Zebra]
  // CHECK: hw.hierpath private @nla [@T_NLATop::@test, @T_Aardvark::@test, @T_A_Z_Zebra]
  // CHECK: hw.hierpath private @nla_1 [@T_NLATop::@test, @T_Aardvark::@test_1, @T_A_Z_Zebra]
  // CHECK: firrtl.module @T_NLATop
  firrtl.module @NLATop()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = true
    }]} {

    // CHECK:  firrtl.instance test sym @test @T_Aardvark()
    firrtl.instance test  sym @test @Aardvark()

    // CHECK: firrtl.instance test2 @T_Z_Zebra()
    firrtl.instance test2 @Zebra()
  }

  // CHECK: firrtl.module @T_Aardvark
  firrtl.module @Aardvark()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "A_",
      inclusive = false
    }]} {

    // CHECK:  firrtl.instance test sym @test @T_A_Z_Zebra()
    firrtl.instance test sym @test @Zebra()
    firrtl.instance test1 sym @test_1 @Zebra()
  }

  // CHECK: firrtl.module @T_Z_Zebra
  // CHECK: firrtl.module @T_A_Z_Zebra
  firrtl.module @Zebra()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "Z_",
      inclusive = true
    }]} {
  }
}

// Prefixes should be applied to Grand Central Data or Mem taps.  Check that a
// multiply instantiated Data/Mem tap is cloned ("duplicated" in Scala FIRRTL
// Compiler terminology) if needed.  (Note: multiply instantiated taps are
// completely untrodden territory for Grand Central.  However, the behavior here
// is the exact same as how normal modules are cloned.)
//
// CHECK-LABEL: firrtl.circuit "GCTDataMemTapsPrefix"
firrtl.circuit "GCTDataMemTapsPrefix" {
  // CHECK:      firrtl.extmodule @FOO_DataTap
  // CHECK-SAME:   defname = "FOO_DataTap"
  firrtl.extmodule @DataTap()
    attributes {annotations = [{
      class = "sifive.enterprise.grandcentral.DataTapsAnnotation.blackbox"}],
      defname = "DataTap"}
  // The Mem tap should be prefixed with "FOO_" and cloned to create a copy
  // prefixed with "BAR_".
  //
  // CHECK:      firrtl.extmodule @FOO_MemTap
  // CHECK-SAME:   defname = "FOO_MemTap"
  // CHECK:      firrtl.extmodule @BAR_MemTap
  // CHECK-SAME:   defname = "BAR_MemTap"
  firrtl.extmodule @MemTap(
    out mem: !firrtl.vector<uint<1>, 1>
      [{
        circt.fieldID = 1 : i32,
        class = "sifive.enterprise.grandcentral.MemTapAnnotation.port",
        id = 0 : i64,
        word = 0 : i64}])
    attributes {defname = "MemTap"}
  // Module DUT has a "FOO_" prefix.
  firrtl.module @DUT()
    attributes {annotations = [
      {class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
       prefix = "FOO_",
       inclusive = true}]} {
    // CHECK: firrtl.instance d @FOO_DataTap
    firrtl.instance d @DataTap()
    // CHECK: firrtl.instance m @FOO_MemTap
    %a = firrtl.instance m @MemTap(out mem: !firrtl.vector<uint<1>, 1>)
  }
  // Module DUT2 has a "BAR_" prefix.
  firrtl.module @DUT2()
    attributes {annotations = [
      {class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
       prefix = "BAR_",
       inclusive = true}]} {
    // CHECK: firrtl.instance m @BAR_MemTap
    %a = firrtl.instance m @MemTap(out mem: !firrtl.vector<uint<1>, 1>)
  }
  firrtl.module @GCTDataMemTapsPrefix() {
    firrtl.instance dut @DUT()
    firrtl.instance dut @DUT2()
  }
}

// Test the NonLocalAnchor is properly updated.
// CHECK-LABEL: firrtl.circuit "FixNLA" {
firrtl.circuit "FixNLA"   {
  hw.hierpath private @nla_1 [@FixNLA::@bar, @Bar::@baz, @Baz]
  // CHECK:   hw.hierpath private @nla_1 [@FixNLA::@bar, @Bar::@baz, @Baz]
  hw.hierpath private @nla_2 [@FixNLA::@foo, @Foo::@bar, @Bar::@baz, @Baz::@s1]
  // CHECK:   hw.hierpath private @nla_2 [@FixNLA::@foo, @X_Foo::@bar, @X_Bar::@baz, @X_Baz::@s1]
  hw.hierpath private @nla_3 [@FixNLA::@bar, @Bar::@baz, @Baz]
  // CHECK:   hw.hierpath private @nla_3 [@FixNLA::@bar, @Bar::@baz, @Baz]
  hw.hierpath private @nla_4 [@Foo::@bar, @Bar::@baz, @Baz]
  // CHECK:       hw.hierpath private @nla_4 [@X_Foo::@bar, @X_Bar::@baz, @X_Baz]
  // CHECK-LABEL: firrtl.module @FixNLA()
  firrtl.module @FixNLA() {
    firrtl.instance foo sym @foo  @Foo()
    firrtl.instance bar sym @bar  @Bar()
    // CHECK:   firrtl.instance foo sym @foo @X_Foo()
    // CHECK:   firrtl.instance bar sym @bar @Bar()
  }
  firrtl.module @Foo() attributes {annotations = [{class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation", inclusive = true, prefix = "X_"}]} {
    firrtl.instance bar sym @bar  @Bar()
  }
  // CHECK-LABEL:   firrtl.module @X_Foo()
  // CHECK:         firrtl.instance bar sym @bar @X_Bar()

  // CHECK-LABEL:   firrtl.module @Bar()
  firrtl.module @Bar() {
    firrtl.instance baz sym @baz @Baz()
    // CHECK:     firrtl.instance baz sym @baz @Baz()
  }
  // CHECK-LABEL: firrtl.module @X_Bar()
  // CHECK:       firrtl.instance baz sym @baz @X_Baz()

  firrtl.module @Baz() attributes {annotations = [{circt.nonlocal = @nla_1, class = "nla_1"}, {circt.nonlocal = @nla_3, class = "nla_3"}, {circt.nonlocal = @nla_4, class = "nla_4"}]} {
    %mem_MPORT_en = firrtl.wire sym @s1  {annotations = [{circt.nonlocal = @nla_2, class = "nla_2"}]} : !firrtl.uint<1>
  }
  // CHECK-LABEL: firrtl.module @X_Baz()
  // CHECK-SAME:  annotations = [{circt.nonlocal = @nla_4, class = "nla_4"}]
  // CHECK:       %mem_MPORT_en = firrtl.wire sym @s1  {annotations = [{circt.nonlocal = @nla_2, class = "nla_2"}]} : !firrtl.uint<1>
  // CHECK:       firrtl.module @Baz()
  // CHECK-SAME:  annotations = [{circt.nonlocal = @nla_1, class = "nla_1"}, {circt.nonlocal = @nla_3, class = "nla_3"}]
  // CHECK:       %mem_MPORT_en = firrtl.wire sym @s1  : !firrtl.uint<1>
}

// Test that NonLocalAnchors are properly updated with memmodules.
firrtl.circuit "Test"   {
  // CHECK: hw.hierpath private @nla_1 [@Test::@foo1, @A_Foo1::@bar, @A_Bar]
  hw.hierpath private @nla_1 [@Test::@foo1, @Foo1::@bar, @Bar]
  // CHECK: hw.hierpath private @nla_2 [@Test::@foo2, @B_Foo2::@bar, @B_Bar]
  hw.hierpath private @nla_2 [@Test::@foo2, @Foo2::@bar, @Bar]

  firrtl.module @Test() {
    firrtl.instance foo1 sym @foo1 @Foo1()
    firrtl.instance foo2 sym @foo2 @Foo2()
  }

  firrtl.module @Foo1() attributes {annotations = [{class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation", inclusive = true, prefix = "A_"}]} {
    firrtl.instance bar sym @bar @Bar()
  }

  firrtl.module @Foo2() attributes {annotations = [{class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation", inclusive = true, prefix = "B_"}]} {
    firrtl.instance bar sym @bar @Bar()
  }

  // CHECK: firrtl.memmodule @A_Bar() attributes {annotations = [{circt.nonlocal = @nla_1, class = "test1"}]
  // CHECK: firrtl.memmodule @B_Bar() attributes {annotations = [{circt.nonlocal = @nla_2, class = "test2"}]
  firrtl.memmodule @Bar() attributes {annotations = [{circt.nonlocal = @nla_1, class = "test1"}, {circt.nonlocal = @nla_2, class = "test2"}], dataWidth = 1 : ui32, depth = 16 : ui64, extraPorts = [], maskBits = 0 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 0 : ui32, readLatency = 0 : ui32,  writeLatency = 1 : ui32}
}

// Test that the MarkDUTAnnotation receives a prefix.
// CHECK-LABEL: firrtl.circuit "Prefix_MarkDUTAnnotationGetsPrefix"
firrtl.circuit "MarkDUTAnnotationGetsPrefix" {
  // CHECK-NEXT: firrtl.module @Prefix_MarkDUTAnnotationGetsPrefix
  // CHECK-SAME:   class = "sifive.enterprise.firrtl.MarkDUTAnnotation", prefix = "Prefix_"
  firrtl.module @MarkDUTAnnotationGetsPrefix() attributes {
    annotations = [
     {
       class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
     },
     {
       class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
       prefix = "Prefix_",
       inclusive = true
     }
    ]
  } {}
}


// Test that inner name refs are properly adjusted.
firrtl.circuit "RewriteInnerNameRefs" {
  // CHECK-LABEL: firrtl.module @Prefix_RewriteInnerNameRefs
  firrtl.module @RewriteInnerNameRefs() attributes {
    annotations = [
     {
       class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
       prefix = "Prefix_",
       inclusive = true
     }
    ]
  } {
    %wire = firrtl.wire sym @wire : !firrtl.uint<1>
    firrtl.instance nested @Nested()

    // CHECK: #hw.innerNameRef<@Prefix_RewriteInnerNameRefs::@wire>
    sv.verbatim "{{0}}" {symbols=[#hw.innerNameRef<@RewriteInnerNameRefs::@wire>]}

    // CHECK: #hw.innerNameRef<@Prefix_RewriteInnerNameRefs::@wire>
    // CHECK: #hw.innerNameRef<@Prefix_Nested::@wire>
    sv.verbatim "{{0}} {{1}}" {symbols=[
      #hw.innerNameRef<@RewriteInnerNameRefs::@wire>,
      #hw.innerNameRef<@Nested::@wire>
    ]}
  }

  firrtl.module @Nested() {
    %wire = firrtl.wire sym @wire : !firrtl.uint<1>
  }
}

// RUN: circt-opt --firrtl-emit-metadata="repl-seq-mem=true repl-seq-mem-file='dut.conf'" -split-input-file %s | FileCheck %s

firrtl.circuit "empty" {
  firrtl.module @empty() {
  }
}
// CHECK-LABEL: firrtl.circuit "empty"   {
// CHECK-NEXT:    firrtl.module @empty() {
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// Memory metadata om class should not be created.
// CHECK-NOT: om.class @MemorySchema

// -----

//===----------------------------------------------------------------------===//
// RetimeModules
//===----------------------------------------------------------------------===//

firrtl.circuit "retime0" attributes { annotations = [{
    class = "sifive.enterprise.firrtl.RetimeModulesAnnotation",
    filename = "retime_modules.json"
}]} {

  firrtl.module @retime0() attributes { annotations = [{
      class = "freechips.rocketchip.util.RetimeModuleAnnotation"
  }]} { }

  firrtl.module @retime1() { }

  firrtl.module @retime2() attributes { annotations = [{
      class = "freechips.rocketchip.util.RetimeModuleAnnotation"
  }]} { }
}
// CHECK-LABEL: firrtl.circuit "retime0"   {
// CHECK:         firrtl.module @retime0() {
// CHECK:         firrtl.module @retime1() {
// CHECK:         firrtl.module @retime2() {
// CHECK{LITERAL}:  sv.verbatim "[\0A \22{{0}}\22,\0A \22{{1}}\22\0A]"
// CHECK-SAME:        output_file = #hw.output_file<"retime_modules.json", excludeFromFileList>
// CHECK-SAME:        symbols = [@retime0, @retime2]

// CHECK:   om.class @RetimeModulesSchema(%moduleName: !om.sym_ref) {
// CHECK-NEXT:     om.class.field @moduleName, %moduleName : !om.sym_ref
// CHECK-NEXT:   }

// CHECK:   om.class @RetimeModulesMetadata() {
// CHECK-NEXT:     %0 = om.constant #om.sym_ref<@retime0> : !om.sym_ref
// CHECK-NEXT:     %1 = om.object @RetimeModulesSchema(%0) : (!om.sym_ref) -> !om.class.type<@RetimeModulesSchema>
// CHECK-NEXT:     om.class.field @[[m1:.+]], %1 : !om.class.type<@RetimeModulesSchema>
// CHECK-NEXT:     %2 = om.constant #om.sym_ref<@retime2> : !om.sym_ref
// CHECK-NEXT:     %3 = om.object @RetimeModulesSchema(%2) : (!om.sym_ref) -> !om.class.type<@RetimeModulesSchema>
// CHECK-NEXT:     om.class.field @[[m2:.+]], %3 : !om.class.type<@RetimeModulesSchema>
// CHECK-NEXT:   }

// -----

//===----------------------------------------------------------------------===//
// SitestBlackbox
//===----------------------------------------------------------------------===//

// CHECK-LABEL: firrtl.circuit "DUTBlackboxes" {
firrtl.circuit "DUTBlackboxes" attributes { annotations = [{
    class = "sifive.enterprise.firrtl.SitestBlackBoxAnnotation",
    filename = "dut_blackboxes.json"
  }]} {
  firrtl.module @DUTBlackboxes() attributes {annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
  }
// CHECK-NOT: sv.verbatim "[]" {output_file = #hw.output_file<"", excludeFromFileList>}
// CHECK:     sv.verbatim "[]" {output_file = #hw.output_file<"dut_blackboxes.json", excludeFromFileList>}
// CHECK-NOT: sv.verbatim "[]" {output_file = #hw.output_file<"", excludeFromFileList>}
}

// -----

// CHECK-LABEL: firrtl.circuit "TestBlackboxes"  {
firrtl.circuit "TestBlackboxes" attributes { annotations = [{
    class = "sifive.enterprise.firrtl.SitestTestHarnessBlackBoxAnnotation",
    filename = "test_blackboxes.json"
  }]} {
  firrtl.module @TestBlackboxes() attributes {annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
  }
// CHECK-NOT: sv.verbatim "[]" {output_file = #hw.output_file<"", excludeFromFileList>}
// CHECK:     sv.verbatim "[]" {output_file = #hw.output_file<"test_blackboxes.json", excludeFromFileList>}
// CHECK-NOT: sv.verbatim "[]" {output_file = #hw.output_file<"", excludeFromFileList>}
}

// -----

// CHECK-LABEL: firrtl.circuit "BasicBlackboxes"   {
firrtl.circuit "BasicBlackboxes" attributes { annotations = [{
    class = "sifive.enterprise.firrtl.SitestBlackBoxAnnotation",
    filename = "dut_blackboxes.json"
  }, {
    class = "sifive.enterprise.firrtl.SitestTestHarnessBlackBoxAnnotation",
    filename = "test_blackboxes.json"
  }]} {

  firrtl.module @BasicBlackboxes() attributes {annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    firrtl.instance test @DUTBlackbox_0()
    firrtl.instance test @DUTBlackbox_1()
    firrtl.instance test @DUTBlackbox_2()
  }

  // These should all be ignored.
  firrtl.extmodule @ignored0() attributes {annotations = [{class = "firrtl.transforms.BlackBoxInlineAnno"}], defname = "ignored0"}
  firrtl.extmodule @ignored1() attributes {annotations = [{class = "firrtl.transforms.BlackBoxPathAnno"}], defname = "ignored1"}
  firrtl.extmodule @ignored2() attributes {annotations = [{class = "sifive.enterprise.firrtl.ScalaClassAnnotation", className = "freechips.rocketchip.util.BlackBoxedROM"}], defname = "ignored2"}
  firrtl.extmodule @ignored3() attributes {annotations = [{class = "sifive.enterprise.grandcentral.DataTapsAnnotation.blackbox"}], defname = "ignored3"}
  firrtl.extmodule @ignored4() attributes {annotations = [{class = "sifive.enterprise.grandcentral.MemTapAnnotation.blackbox", id = 4 : i64}], defname = "ignored4"}
  firrtl.extmodule @ignored5() attributes {annotations = [{class = "firrtl.transforms.BlackBox"}], defname = "ignored5"}

  // ScalaClassAnnotation should be discarded after this pass.
  // CHECK: firrtl.extmodule @ignored2()
  // CHECK-NOT: sifive.enterprise.firrtl.ScalaClassAnnotation

  // Gracefully handle missing defnames.
  firrtl.extmodule @NoDefName()

  firrtl.extmodule @TestBlackbox() attributes {defname = "TestBlackbox"}
  // CHECK: sv.verbatim "[\0A \22TestBlackbox\22\0A]" {output_file = #hw.output_file<"test_blackboxes.json", excludeFromFileList>}

  // Should be de-duplicated and sorted.
  firrtl.extmodule @DUTBlackbox_0() attributes {defname = "DUTBlackbox2"}
  firrtl.extmodule @DUTBlackbox_1() attributes {defname = "DUTBlackbox1"}
  firrtl.extmodule @DUTBlackbox_2() attributes {defname = "DUTBlackbox1"}
  // CHECK: sv.verbatim "[\0A \22DUTBlackbox1\22,\0A \22DUTBlackbox2\22\0A]" {output_file = #hw.output_file<"dut_blackboxes.json", excludeFromFileList>}
}
// CHECK:  om.class @SitestBlackBoxModulesSchema(%moduleName: !om.sym_ref) {
// CHECK-NEXT:    om.class.field @moduleName, %moduleName : !om.sym_ref
// CHECK-NEXT:  }

// CHECK:   om.class @SitestBlackBoxMetadata() {
// CHECK:     %0 = om.constant #om.sym_ref<@TestBlackbox> : !om.sym_ref
// CHECK:     %1 = om.object @SitestBlackBoxModulesSchema(%0)
// CHECK:     om.class.field @exterMod_TestBlackbox, %1
// CHECK:     %2 = om.constant #om.sym_ref<@DUTBlackbox_0> : !om.sym_ref
// CHECK:     %3 = om.object @SitestBlackBoxModulesSchema(%2)
// CHECK:     om.class.field @exterMod_DUTBlackbox_0, %3
// CHECK:     %4 = om.constant #om.sym_ref<@DUTBlackbox_1> : !om.sym_ref
// CHECK:     %5 = om.object @SitestBlackBoxModulesSchema(%4)
// CHECK:     om.class.field @exterMod_DUTBlackbox_1, %5
// CHECK:     %6 = om.constant #om.sym_ref<@DUTBlackbox_2> : !om.sym_ref
// CHECK:     %7 = om.object @SitestBlackBoxModulesSchema(%6)
// CHECK:     om.class.field @exterMod_DUTBlackbox_2, %7
// CHECK:   }

// -----

//===----------------------------------------------------------------------===//
// MemoryMetadata
//===----------------------------------------------------------------------===//

// CHECK-LABEL: firrtl.circuit "top"
firrtl.circuit "top"
{
  firrtl.module @top() { }
  // When there are no memories, we still need to emit the memory metadata.
  // CHECK: sv.verbatim "[]" {output_file = #hw.output_file<"metadata{{/|\\\\}}seq_mems.json", excludeFromFileList>}
  // CHECK: sv.verbatim "" {output_file = #hw.output_file<"'dut.conf'", excludeFromFileList>}
}

// -----

// CHECK-LABEL: firrtl.circuit "OneMemory"
firrtl.circuit "OneMemory" {
  firrtl.module @OneMemory() {
    %0:5= firrtl.instance MWrite_ext sym @MWrite_ext_0  @MWrite_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>, in user_input: !firrtl.uint<5>)
  }
  firrtl.memmodule @MWrite_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>, in user_input: !firrtl.uint<5>) attributes {dataWidth = 42 : ui32, depth = 12 : ui64, extraPorts = [{direction = "input", name = "user_input", width = 5 : ui32}], maskBits = 1 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
  // CHECK{LITERAL}: "[\0A {\0A \22module_name\22: \22{{0}}\22,\0A \22depth\22: 12,\0A \22width\22: 42,\0A \22masked\22: false,\0A \22read\22: 0,\0A \22write\22: 1,\0A \22readwrite\22: 0,\0A \22extra_ports\22: [\0A {\0A \22name\22: \22user_input\22,\0A \22direction\22: \22input\22,\0A \22width\22: 5\0A }\0A ],\0A \22hierarchy\22: [\0A \22{{1}}.MWrite_ext\22\0A ]\0A }\0A]"
  // CHECK-SAME: symbols = [@MWrite_ext, @OneMemory]
  // CHECK{LITERAL}: sv.verbatim "name {{0}} depth 12 width 42 ports write\0A" {output_file = #hw.output_file<"'dut.conf'"
  // CHECK-SAME: symbols = [@MWrite_ext]
}

// -----

// CHECK-LABEL: firrtl.circuit "DualReadsSMem"
firrtl.circuit "DualReadsSMem" {
  firrtl.module @DualReadsSMem() {
    %0:12 = firrtl.instance DualReads_ext {annotations = [{class = "sifive.enterprise.firrtl.SeqMemInstanceMetadataAnnotation", data = {baseAddress = 2147483648 : i64, dataBits = 8 : i64, eccBits = 0 : i64, eccIndices = [], eccScheme = "none"}}]}  @DualReads_ext(in R0_addr: !firrtl.uint<4>, in R0_en: !firrtl.uint<1>, in R0_clk: !firrtl.clock, in R0_data: !firrtl.uint<42>, in R1_addr: !firrtl.uint<4>, in R1_en: !firrtl.uint<1>, in R1_clk: !firrtl.clock, in R1_data: !firrtl.uint<42>, in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>)
  }
  firrtl.memmodule @DualReads_ext(in R0_addr: !firrtl.uint<4>, in R0_en: !firrtl.uint<1>, in R0_clk: !firrtl.clock, in R0_data: !firrtl.uint<42>, in R1_addr: !firrtl.uint<4>, in R1_en: !firrtl.uint<1>, in R1_clk: !firrtl.clock, in R1_data: !firrtl.uint<42>, in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>) attributes {dataWidth = 42 : ui32, depth = 12 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 2 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
  // CHECK{LITERAL}: sv.verbatim "[\0A {\0A \22module_name\22: \22{{0}}\22,\0A \22depth\22: 12,\0A \22width\22: 42,\0A \22masked\22: false,\0A \22read\22: 2,\0A \22write\22: 1,\0A \22readwrite\22: 0,\0A \22extra_ports\22: [],\0A \22hierarchy\22: [\0A \22{{1}}.DualReads_ext\22\0A ]\0A }\0A]"
  // CHECK: symbols = [@DualReads_ext, @DualReadsSMem]}
  // CHECK{LITERAL}: sv.verbatim "name {{0}} depth 12 width 42 ports write,read,read\0A" {output_file = #hw.output_file<"'dut.conf'", excludeFromFileList>, symbols = [@DualReads_ext]}
}

// -----

// CHECK-LABEL: firrtl.circuit "top"
firrtl.circuit "top" {
    firrtl.module @top()  {
      // CHECK: firrtl.instance dut sym @[[DUT_SYM:.+]] @DUT
      firrtl.instance dut @DUT()
      firrtl.instance mem1 @Mem1()
      firrtl.instance mem2 @Mem2()
    }
    firrtl.module private @Mem1() {
      %0:4 = firrtl.instance head_ext  @head_ext(in W0_addr: !firrtl.uint<5>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<5>)
    }
    firrtl.module private @Mem2() {
      %0:4 =  firrtl.instance head_0_ext  @head_0_ext(in W0_addr: !firrtl.uint<5>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<5>)
    }
    firrtl.module private @DUT() attributes {annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
      // CHECK: firrtl.instance mem1 sym @[[MEM1_SYM:.+]] @Mem(
      firrtl.instance mem1 @Mem()
    }
    firrtl.module private @Mem() {
      %0:10 = firrtl.instance memory_ext {annotations = [{class = "sifive.enterprise.firrtl.SeqMemInstanceMetadataAnnotation", data = {baseAddress = 2147483648 : i64, dataBits = 8 : i64, eccBits = 0 : i64, eccIndices = [], eccScheme = "none"}}]} @memory_ext(in R0_addr: !firrtl.uint<4>, in R0_en: !firrtl.uint<1>, in R0_clk: !firrtl.clock, out R0_data: !firrtl.uint<8>, in RW0_addr: !firrtl.uint<4>, in RW0_en: !firrtl.uint<1>, in RW0_clk: !firrtl.clock, in RW0_wmode: !firrtl.uint<1>, in RW0_wdata: !firrtl.uint<8>, out RW0_rdata: !firrtl.uint<8>)
      %1:8 = firrtl.instance dumm_ext @dumm_ext(in R0_addr: !firrtl.uint<5>, in R0_en: !firrtl.uint<1>, in R0_clk: !firrtl.clock, out R0_data: !firrtl.uint<5>, in W0_addr: !firrtl.uint<5>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<5>)
    }
    firrtl.memmodule private @head_ext(in W0_addr: !firrtl.uint<5>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<5>) attributes {dataWidth = 5 : ui32, depth = 20 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
    firrtl.memmodule private @head_0_ext(in W0_addr: !firrtl.uint<5>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<5>) attributes {dataWidth = 5 : ui32, depth = 20 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
    firrtl.memmodule private @memory_ext(in R0_addr: !firrtl.uint<4>, in R0_en: !firrtl.uint<1>, in R0_clk: !firrtl.clock, out R0_data: !firrtl.uint<8>, in RW0_addr: !firrtl.uint<4>, in RW0_en: !firrtl.uint<1>, in RW0_clk: !firrtl.clock, in RW0_wmode: !firrtl.uint<1>, in RW0_wdata: !firrtl.uint<8>, out RW0_rdata: !firrtl.uint<8>) attributes {dataWidth = 8 : ui32, depth = 16 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 1 : ui32, numWritePorts = 0 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
    firrtl.memmodule private @dumm_ext(in R0_addr: !firrtl.uint<5>, in R0_en: !firrtl.uint<1>, in R0_clk: !firrtl.clock, out R0_data: !firrtl.uint<5>, in W0_addr: !firrtl.uint<5>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<5>) attributes {dataWidth = 5 : ui32, depth = 20 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
    // CHECK{LITERAL}: sv.verbatim "[\0A {\0A \22module_name\22: \22{{0}}\22,\0A \22depth\22: 16,\0A \22width\22: 8,\0A \22masked\22: false,\0A \22read\22: 1,\0A \22write\22: 0,\0A \22readwrite\22: 1,\0A \22extra_ports\22: [],\0A \22hierarchy\22: [\0A \22{{3}}.{{4}}.memory_ext\22\0A ]\0A },\0A {\0A \22module_name\22: \22{{5}}\22,\0A \22depth\22: 20,\0A \22width\22: 5,\0A \22masked\22: false,\0A \22read\22: 1,\0A \22write\22: 1,\0A \22readwrite\22: 0,\0A \22extra_ports\22: [],\0A \22hierarchy\22: [\0A \22{{3}}.{{4}}.dumm_ext\22\0A ]\0A }\0A]"
    // CHECK-SAME: symbols = [@memory_ext, @top, #hw.innerNameRef<@top::@[[DUT_SYM]]>, @DUT, #hw.innerNameRef<@DUT::@[[MEM1_SYM]]>, @dumm_ext]
    // CHECK{LITERAL}: sv.verbatim "name {{0}} depth 20 width 5 ports write\0Aname {{1}} depth 20 width 5 ports write\0Aname {{2}} depth 16 width 8 ports read,rw\0Aname {{3}} depth 20 width 5 ports write,read\0A"
    // CHECK-SAME: {output_file = #hw.output_file<"'dut.conf'", excludeFromFileList
    // CHECK-SAME: symbols = [@head_ext, @head_0_ext, @memory_ext, @dumm_ext]
}

// CHECK:  om.class @MemorySchema(%name: !om.sym_ref, %depth: ui64, %width: ui32, %maskBits: ui32, %readPorts: ui32, %writePorts: ui32, %readwritePorts: ui32, %writeLatency: ui32, %readLatency: ui32) {
// CHECK-NEXT:    om.class.field @name, %name : !om.sym_ref
// CHECK-NEXT:    om.class.field @depth, %depth : ui64
// CHECK-NEXT:    om.class.field @width, %width : ui32
// CHECK-NEXT:    om.class.field @maskBits, %maskBits : ui32
// CHECK-NEXT:    om.class.field @readPorts, %readPorts : ui32
// CHECK-NEXT:    om.class.field @writePorts, %writePorts : ui32
// CHECK-NEXT:    om.class.field @readwritePorts, %readwritePorts : ui32
// CHECK-NEXT:    om.class.field @writeLatency, %writeLatency : ui32
// CHECK-NEXT:    om.class.field @readLatency, %readLatency : ui32
// CHECK-NEXT:  }

// CHECK:  om.class @MemoryMetadata() {
// CHECK-NEXT:    %[[v0:.+]] = om.constant #om.sym_ref<@head_ext> : !om.sym_ref
// CHECK-NEXT:    %[[v1:.+]] = om.constant 20 : ui64
// CHECK-NEXT:    %[[v2:.+]] = om.constant 5 : ui32
// CHECK-NEXT:    %[[v3:.+]] = om.constant 1 : ui32
// CHECK-NEXT:    %[[v4:.+]] = om.constant 0 : ui32
// CHECK-NEXT:    %[[v5:.+]] = om.constant 1 : ui32
// CHECK-NEXT:    %[[v6:.+]] = om.constant 0 : ui32
// CHECK-NEXT:    %[[v7:.+]] = om.constant 1 : ui32
// CHECK-NEXT:    %[[v8:.+]] = om.constant 1 : ui32
// CHECK-NEXT:    %[[v9:.+]] = om.object @MemorySchema(%[[v0]], %[[v1]], %[[v2]], %[[v3]], %[[v4]], %[[v5]], %[[v6]], %[[v7]], %[[v8]])
// CHECK-NEXT:    om.class.field @[[m0:.+]], %[[v9]] : !om.class.type<@MemorySchema>
// CHECK-NEXT:    %[[v10:.+]] = om.constant #om.sym_ref<@head_0_ext> : !om.sym_ref
// CHECK-NEXT:    %[[v11:.+]] = om.constant 20 : ui64
// CHECK-NEXT:    %[[v12:.+]] = om.constant 5 : ui32
// CHECK-NEXT:    %[[v13:.+]] = om.constant 1 : ui32
// CHECK-NEXT:    %[[v14:.+]] = om.constant 0 : ui32
// CHECK-NEXT:    %[[v15:.+]] = om.constant 1 : ui32
// CHECK-NEXT:    %[[v16:.+]] = om.constant 0 : ui32
// CHECK-NEXT:    %[[v17:.+]] = om.constant 1 : ui32
// CHECK-NEXT:    %[[v18:.+]] = om.constant 1 : ui32
// CHECK-NEXT:    %[[v19:.+]] = om.object @MemorySchema(%[[v10]], %[[v11]], %[[v12]], %[[v13]], %[[v14]], %[[v15]], %[[v16]], %[[v17]], %[[v18]])
// CHECK-NEXT:    om.class.field @[[m1:.+]], %[[v19]] : !om.class.type<@MemorySchema>
// CHECK-NEXT:    %[[v20:.+]] = om.constant #om.sym_ref<@memory_ext> : !om.sym_ref
// CHECK-NEXT:    %[[v21:.+]] = om.constant 16 : ui64
// CHECK-NEXT:    %[[v22:.+]] = om.constant 8 : ui32
// CHECK-NEXT:    %[[v23:.+]] = om.constant 1 : ui32
// CHECK-NEXT:    %[[v24:.+]] = om.constant 1 : ui32
// CHECK-NEXT:    %[[v25:.+]] = om.constant 0 : ui32
// CHECK-NEXT:    %[[v26:.+]] = om.constant 1 : ui32
// CHECK-NEXT:    %[[v27:.+]] = om.constant 1 : ui32
// CHECK-NEXT:    %[[v28:.+]] = om.constant 1 : ui32
// CHECK-NEXT:    %[[v29:.+]] = om.object @MemorySchema(%[[v20]], %[[v21]], %[[v22]], %[[v23]], %[[v24]], %[[v25]], %[[v26]], %[[v27]], %[[v28]])
// CHECK-NEXT:    om.class.field @[[m2:.+]], %[[v29]] : !om.class.type<@MemorySchema>
// CHECK-NEXT:    %[[v30:.+]] = om.constant #om.sym_ref<@dumm_ext> : !om.sym_ref
// CHECK-NEXT:    %[[v31:.+]] = om.constant 20 : ui64
// CHECK-NEXT:    %[[v32:.+]] = om.constant 5 : ui32
// CHECK-NEXT:    %[[v33:.+]] = om.constant 1 : ui32
// CHECK-NEXT:    %[[v34:.+]] = om.constant 1 : ui32
// CHECK-NEXT:    %[[v35:.+]] = om.constant 1 : ui32
// CHECK-NEXT:    %[[v36:.+]] = om.constant 0 : ui32
// CHECK-NEXT:    %[[v37:.+]] = om.constant 1 : ui32
// CHECK-NEXT:    %[[v38:.+]] = om.constant 1 : ui32
// CHECK-NEXT:    %[[v39:.+]] = om.object @MemorySchema(%[[v30]], %[[v31]], %[[v32]], %[[v33]], %[[v34]], %[[v35]], %[[v36]], %[[v37]], %[[v38]]) : (!om.sym_ref, ui64, ui32, ui32, ui32, ui32, ui32, ui32, ui32) -> !om.class.type<@MemorySchema>
// CHECK-NEXT:    om.class.field @[[m3:.+]], %[[v39]] : !om.class.type<@MemorySchema>
// CHECK-NEXT:  }

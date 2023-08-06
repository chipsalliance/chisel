// RUN: circt-opt -hw-memory-sim %s | FileCheck %s --check-prefix COMMON --implicit-check-not sv.attributes
// RUN: circt-opt -pass-pipeline="builtin.module(hw-memory-sim{ignore-read-enable})" %s | FileCheck %s --check-prefixes=COMMON,IGNORE
// RUN: circt-opt -pass-pipeline="builtin.module(hw-memory-sim{add-mux-pragmas})" %s | FileCheck %s --check-prefixes=COMMON,PRAMGAS
// RUN: circt-opt -pass-pipeline="builtin.module(hw-memory-sim{disable-mem-randomization})" %s | FileCheck %s --check-prefix COMMON --implicit-check-not RANDOMIZE_MEM
// RUN: circt-opt -pass-pipeline="builtin.module(hw-memory-sim{disable-reg-randomization})" %s | FileCheck %s --check-prefix COMMON --implicit-check-not RANDOMIZE_REG
// RUN: circt-opt -pass-pipeline="builtin.module(hw-memory-sim{disable-mem-randomization disable-reg-randomization})" %s | FileCheck %s --check-prefix COMMON --implicit-check-not RANDOMIZE_REG --implicit-check-not RANDOMIZE_MEM
// RUN: circt-opt -pass-pipeline="builtin.module(hw-memory-sim{add-vivado-ram-address-conflict-synthesis-bug-workaround})" %s | FileCheck %s --check-prefixes=CHECK,COMMON,VIVADO

hw.generator.schema @FIRRTLMem, "FIRRTL_Memory", ["depth", "numReadPorts", "numWritePorts", "numReadWritePorts", "readLatency", "writeLatency", "width", "readUnderWrite", "writeUnderWrite", "writeClockIDs", "initFilename", "initIsBinary", "initIsInline"]

sv.macro.decl @RANDOM

// COMMON-LABEL: @complex
hw.module @complex(%clock: i1, %reset: i1, %r0en: i1, %mode: i1, %data0: i16) -> (data1: i16, data2: i16) {
  %true = hw.constant true
  %c0_i4 = hw.constant 0 : i4
  %tmp41.ro_data_0, %tmp41.rw_rdata_0 = hw.instance "tmp41"
   @FIRRTLMem_1_1_1_16_10_2_4_0_0(ro_addr_0: %c0_i4: i4, ro_en_0: %r0en: i1,
     ro_clock_0: %clock: i1, rw_addr_0: %c0_i4: i4, rw_en_0: %r0en: i1,
     rw_clock_0: %clock: i1, rw_wmode_0: %mode: i1,
     rw_wdata_0: %data0: i16,
     wo_addr_0: %c0_i4: i4, wo_en_0: %r0en: i1,
     wo_clock_0: %clock: i1, wo_data_0: %data0: i16) -> (ro_data_0: i16, rw_rdata_0: i16)

  hw.output %tmp41.ro_data_0, %tmp41.rw_rdata_0 : i16, i16
}

hw.module @complexMultiBit(%clock: i1, %reset: i1, %r0en: i1, %mode: i1, %data0: i16) -> (data1: i16, data2: i16) {
  %true = hw.constant 1 : i2
  %c0_i4 = hw.constant 0 : i4
  %tmp41.ro_data_0, %tmp41.rw_rdata_0 = hw.instance "tmp41"
   @FIRRTLMem_1_1_1_16_10_2_4_0_0_multi(ro_addr_0: %c0_i4: i4, ro_en_0: %r0en: i1,
     ro_clock_0: %clock: i1, rw_addr_0: %c0_i4: i4, rw_en_0: %r0en: i1,
     rw_clock_0: %clock: i1, rw_wmode_0: %mode: i1,
     rw_wdata_0: %data0: i16,rw_wmask_0: %true: i2,
     wo_addr_0: %c0_i4: i4, wo_en_0: %r0en: i1,
     wo_clock_0: %clock: i1, wo_data_0: %data0: i16, wo_mask_0: %true: i2) -> (ro_data_0: i16, rw_rdata_0: i16)

  hw.output %tmp41.ro_data_0, %tmp41.rw_rdata_0 : i16, i16
}
// COMMON-LABEL: @simple
hw.module @simple(%clock: i1, %reset: i1, %r0en: i1, %mode: i1, %data0: i16) -> (data1: i16, data2: i16) {
  %true = hw.constant true
  %c0_i4 = hw.constant 0 : i4
  %tmp41.ro_data_0, %tmp41.rw_rdata_0 = hw.instance "tmp41"
   @FIRRTLMem_1_1_1_16_10_0_1_0_0( ro_addr_0: %c0_i4: i4,ro_en_0: %r0en: i1,
     ro_clock_0: %clock: i1, rw_addr_0: %c0_i4: i4, rw_en_0: %r0en: i1,
     rw_clock_0: %clock: i1, rw_wmode_0: %mode: i1,
     rw_wdata_0: %data0: i16,
     wo_addr_0: %c0_i4: i4, wo_en_0: %r0en: i1,
     wo_clock_0: %clock: i1, wo_data_0: %data0: i16) ->
     (ro_data_0: i16, rw_rdata_0: i16)

  hw.output %tmp41.ro_data_0, %tmp41.rw_rdata_0 : i16, i16
}

// COMMON-LABEL: @WriteOrderedSameClock
hw.module @WriteOrderedSameClock(%clock: i1, %w0_addr: i4, %w0_en: i1, %w0_data: i8, %w0_mask: i1, %w1_addr: i4, %w1_en: i1, %w1_data: i8, %w1_mask: i1) {
  hw.instance "memory"
    @FIRRTLMemOneAlways(wo_addr_0: %w0_addr: i4, wo_en_0: %w0_en: i1,
      wo_clock_0: %clock: i1, wo_data_0: %w0_data: i8,
      wo_addr_1: %w1_addr: i4, wo_en_1: %w1_en: i1, wo_clock_1: %clock: i1,
       wo_data_1: %w1_data: i8) -> ()
  hw.output
}

// COMMON-LABEL: @WriteOrderedDifferentClock
hw.module @WriteOrderedDifferentClock(%clock: i1, %clock2: i1, %w0_addr: i4, %w0_en: i1, %w0_data: i8, %w0_mask: i1, %w1_addr: i4, %w1_en: i1, %w1_data: i8, %w1_mask: i1) {
  hw.instance "memory"
    @FIRRTLMemTwoAlways(wo_addr_0: %w0_addr: i4, wo_en_0: %w0_en: i1,
      wo_clock_0: %clock: i1, wo_data_0: %w0_data: i8,
      wo_addr_1: %w1_addr: i4, wo_en_1: %w1_en: i1, wo_clock_1: %clock2: i1,
      wo_data_1: %w1_data: i8) -> ()
  hw.output
}

hw.module.generated @FIRRTLMem_1_1_1_16_10_0_1_0_0, @FIRRTLMem(%ro_addr_0: i4, %ro_en_0: i1, %ro_clock_0: i1,%rw_addr_0: i4, %rw_en_0: i1,  %rw_clock_0: i1, %rw_wmode_0: i1, %rw_wdata_0: i16,  %wo_addr_0: i4, %wo_en_0: i1, %wo_clock_0: i1, %wo_data_0: i16) -> (ro_data_0: i16, rw_rdata_0: i16) attributes {depth = 10 : i64, numReadPorts = 1 : ui32, numReadWritePorts = 1 : ui32, numWritePorts = 1 : ui32, readLatency = 0 : ui32, readUnderWrite = 0 : i32, width = 16 : ui32, writeClockIDs = [], writeLatency = 1 : ui32, writeUnderWrite = 0 : i32, initFilename = "", initIsBinary = false, initIsInline = false}

//COMMON-LABEL: @FIRRTLMem_1_1_1_16_10_0_1_0_0
//CHECK-SAME:  attributes {comment = "VCS coverage exclude_file"}
//CHECK:       %Memory = sv.reg
//VIVADO-SAME:  #sv.attribute<"ram_style" = "\22distributed\22">
//CHECK-SAME:  !hw.inout<uarray<10xi16>>
//CHECK-NEXT:  %[[rslot:.+]] = sv.array_index_inout %Memory[%ro_addr_0]
//CHECK-NEXT:  %[[read:.+]] = sv.read_inout %[[rslot]]
//CHECK-NEXT:  %[[x:.+]] = sv.constantX
//CHECK-NEXT:  %[[readres:.+]] = comb.mux %ro_en_0, %[[read]], %[[x]]
//CHECK-NEXT:  %true = hw.constant true
//CHECK-NEXT:  %[[rwtmp:.+]] = sv.wire
//CHECK-NEXT:  %[[rwres:.+]] = sv.read_inout %[[rwtmp]]
//CHECK-NEXT:  %false = hw.constant false
//CHECK-NEXT:  %[[rwrcondpre:.+]] = comb.icmp eq %rw_wmode_0, %false
//CHECK-NEXT:  %[[rwrcond:.+]] = comb.and %rw_en_0, %[[rwrcondpre]]
//CHECK-NEXT:  %[[rwrslot:.+]] = sv.array_index_inout %Memory[%rw_addr_0]
//CHECK-NEXT:  %[[rwdata:.+]] = sv.read_inout %[[rwrslot]]
//CHECK-NEXT:  %[[x2:.+]] = sv.constantX
//CHECK-NEXT:  %[[rwdata2:.+]] = comb.mux %[[rwrcond]], %[[rwdata]], %[[x2]]
//CHECK-NEXT:  sv.assign %[[rwtmp]], %[[rwdata2:.+]]
//CHECK-NEXT:    sv.always posedge %rw_clock_0 {
//CHECK-NEXT:      %[[rwwcondpre:.+]] = comb.and %rw_wmode_0, %true
//CHECK-NEXT:      %[[rwwcond:.+]] = comb.and %rw_en_0, %[[rwwcondpre]]
//CHECK-NEXT:      sv.if %[[rwwcond]]  {
//CHECK-NEXT:        %[[rwwslot:.+]] = sv.array_index_inout %Memory[%rw_addr_0]
//CHECK-NEXT:        %[[c0_i32:.+]] = hw.constant 0 : i32
//CHECK-NEXT:        sv.passign %[[rwwslot]], %rw_wdata_0
//CHECK-NEXT:      }
//CHECK-NEXT:    }
//CHECK-NEXT:  %true_1 = hw.constant true
//CHECK-NEXT:  sv.always posedge %wo_clock_0 {
//CHECK-NEXT:    sv.if %wo_en_0 {
//CHECK-NEXT:      %[[wslot:.+]] = sv.array_index_inout %Memory[%wo_addr_0]
//CHECK-NEXT:      %[[c0_i32:.+]] = hw.constant 0 : i32
//CHECK-NEXT:      sv.passign %[[wslot]], %wo_data_0
//CHECK-NEXT:    }
//CHECK-NEXT:  }
//CHECK-NEXT:  sv.ifdef "ENABLE_INITIAL_MEM_" {
//CHECK-NEXT:    sv.ifdef "RANDOMIZE_REG_INIT" {
//CHECK-NEXT:    }
//CHECK-NEXT:    %_RANDOM_MEM = sv.reg : !hw.inout<i32>
//CHECK-NEXT:    sv.initial {
//CHECK-NEXT:      sv.verbatim "`INIT_RANDOM_PROLOG_"
//CHECK-NEXT:      sv.ifdef.procedural "RANDOMIZE_MEM_INIT" {
//CHECK:             sv.for %i = %c0_i4 to %c-6_i4 step %c1_i4 : i4 {
//CHECK:               sv.for %j = %c0_i6 to %c-32_i6 step %c-32_i6_2 : i6 {
//CHECK:                 %RANDOM = sv.macro.ref.se @RANDOM
//CHECK:                 %[[PART_SELECT:.+]] = sv.indexed_part_select_inout %_RANDOM_MEM[%j : 32] : !hw.inout<i32>, i6
//CHECK:                 sv.bpassign %[[PART_SELECT]], %RANDOM : i32
//CHECK:               }
//CHECK:               %[[MEM_INDEX:.+]] = sv.array_index_inout %Memory[%i] : !hw.inout<uarray<10xi16>>, i4
//CHECK:               %[[READ:.+]] = sv.read_inout %_RANDOM_MEM : !hw.inout<i32>
//CHECK:               %[[EXTRACT:.+]] = comb.extract %[[READ]] from 0 : (i32) -> i16
//CHECK:               sv.bpassign %[[MEM_INDEX]], %[[EXTRACT]] : i16
//CHECK:             }
//CHECK-NEXT:      }
//CHECK-NEXT:      sv.ifdef.procedural "RANDOMIZE_REG_INIT" {
//CHECK-NEXT:      }
//CHECK-NEXT:    }
//CHECK-NEXT:  }
//CHECK-NEXT:  hw.output %[[readres]], %[[rwres]]

// IGNORE: %[[Memory:.+]] = sv.reg : !hw.inout<uarray<10xi16>>
// IGNORE: %[[ro_slot:.+]] = sv.array_index_inout %[[Memory]][%ro_addr_0] : !hw.inout<uarray<10xi16>>, i4
// IGNORE: %[[result_ro:.+]] = sv.read_inout %[[ro_slot]] : !hw.inout<i16>
// IGNORE: %[[rw_wire:.+]] = sv.wire : !hw.inout<i16>
// IGNORE: %[[wire_slot:.+]] = sv.read_inout %2 : !hw.inout<i16>
// IGNORE: %[[rw_slot:.+]] = sv.array_index_inout %[[Memory]][%rw_addr_0] : !hw.inout<uarray<10xi16>>, i4
// IGNORE: %[[rw_value:.+]] = sv.read_inout %[[rw_slot]] : !hw.inout<i16>
// IGNORE: sv.assign %[[rw_wire]], %[[rw_value]] : i16
// IGNORE: hw.output %[[result_ro]], %[[wire_slot]] : i16, i16

hw.module.generated @FIRRTLMem_1_1_1_16_10_2_4_0_0, @FIRRTLMem(%ro_addr_0: i4, %ro_en_0: i1, %ro_clock_0: i1,%rw_addr_0: i4, %rw_en_0: i1,  %rw_clock_0: i1, %rw_wmode_0: i1, %rw_wdata_0: i16, %wo_addr_0: i4, %wo_en_0: i1, %wo_clock_0: i1, %wo_data_0: i16) -> (ro_data_0: i16, rw_rdata_0: i16) attributes {depth = 10 : i64, numReadPorts = 1 : ui32, numReadWritePorts = 1 : ui32, numWritePorts = 1 : ui32, readLatency = 2 : ui32, readUnderWrite = 0 : i32, width = 16 : ui32, writeClockIDs = [], writeLatency = 4 : ui32, writeUnderWrite = 0 : i32, initFilename = "", initIsBinary = false, initIsInline = false}

//COMMON-LABEL: @FIRRTLMem_1_1_1_16_10_2_4_0_0
//COM: This produces a lot of output, we check one field's pipeline
//CHECK:         %Memory = sv.reg
//CHECK-NEXT:    [[EN_0:%.+]] = sv.reg {{.+}} : !hw.inout<i1>
//CHECK-NEXT:    [[EN_1:%.+]] = sv.reg {{.+}} : !hw.inout<i1>
//CHECK-NEXT:    [[ADDR_0:%.+]] = sv.reg {{.+}} : !hw.inout<i4>
//CHECK-NEXT:    [[ADDR_1:%.+]] = sv.reg {{.+}} : !hw.inout<i4>
//CHECK-NEXT:    sv.always posedge %ro_clock_0 {
//CHECK-NEXT:      sv.passign [[EN_0]], %ro_en_0 : i1
//CHECK-NEXT:      [[EN_0R:%.+]] = sv.read_inout [[EN_0]] : !hw.inout<i1>
//CHECK-NEXT:      sv.passign [[EN_1]], [[EN_0R]] : i1
//CHECK-NEXT:      sv.passign [[ADDR_0]], %ro_addr_0 : i4
//CHECK-NEXT:      [[ADDR_0R:%.+]] = sv.read_inout [[ADDR_0]] : !hw.inout<i4>
//CHECK-NEXT:      sv.passign [[ADDR_1]], [[ADDR_0R]] : i4
//CHECK-NEXT:    }
//CHECK-NEXT:    [[EN_1R:%.+]] = sv.read_inout [[EN_1]] : !hw.inout<i1>
//CHECK-NEXT:    [[ADDR_1R:%.+]] = sv.read_inout [[ADDR_1]] : !hw.inout<i4>
//CHECK-NEXT:    {{%.+}} = sv.array_index_inout %Memory[[[ADDR_1R]]] : !hw.inout<uarray<10xi16>>, i4

hw.module.generated @FIRRTLMem_1_1_1_16_1_0_1_0_0, @FIRRTLMem(%ro_addr_0: i4, %ro_en_0: i1, %ro_clock_0: i1,%rw_addr_0: i4, %rw_en_0: i1,  %rw_clock_0: i1, %rw_wmode_0: i1, %rw_wdata_0: i16,  %wo_addr_0: i4, %wo_en_0: i1, %wo_clock_0: i1, %wo_data_0: i16) -> (ro_data_0: i16, rw_rdata_0: i16) attributes {depth = 1 : i64, numReadPorts = 1 : ui32, numReadWritePorts = 1 : ui32, numWritePorts = 1 : ui32, readLatency = 0 : ui32, readUnderWrite = 0 : i32, width = 16 : ui32, writeClockIDs = [], writeLatency = 1 : ui32, writeUnderWrite = 0 : i32, initFilename = "", initIsBinary = false, initIsInline = false}
// COMMON-LABEL: @FIRRTLMem_1_1_1_16_1_0_1_0_0
// CHECK-NOT: infer_mux_override

hw.module.generated @FIRRTLMemOneAlways, @FIRRTLMem( %wo_addr_0: i4, %wo_en_0: i1, %wo_clock_0: i1,%wo_data_0: i8, %wo_addr_1: i4,  %wo_en_1: i1, %wo_clock_1: i1, %wo_data_1: i8) attributes {depth = 16 : i64, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 2 : ui32, readLatency = 1 : ui32, readUnderWrite = 0 : i32, width = 8 : ui32, writeClockIDs = [0 : i32, 0 : i32], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32, initFilename = "", initIsBinary = false, initIsInline = false}

//COMMON-LABEL: @FIRRTLMemOneAlways
//CHECK-COUNT-1:  sv.always
//CHECK-NOT:      sv.always

hw.module.generated @FIRRTLMemTwoAlways, @FIRRTLMem( %wo_addr_0: i4, %wo_en_0: i1, %wo_clock_0: i1,%wo_data_0: i8, %wo_addr_1: i4,  %wo_en_1: i1, %wo_clock_1: i1, %wo_data_1: i8) attributes {depth = 16 : i64, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 2 : ui32, readLatency = 1 : ui32, readUnderWrite = 0 : i32, width = 8 : ui32, writeClockIDs = [0 : i32, 1 : i32], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32, initFilename = "", initIsBinary = false, initIsInline = false}

//COMMON-LABEL: @FIRRTLMemTwoAlways
//CHECK-COUNT-2:  sv.always
//CHECK-NOT:      sv.always


  hw.module.generated @FIRRTLMem_1_1_0_32_16_1_1_0_1_a, @FIRRTLMem(%R0_addr: i4, %R0_en: i1, %R0_clk: i1, %W0_addr: i4, %W0_en: i1, %W0_clk: i1, %W0_data: i32, %W0_mask: i4) -> (R0_data: i32) attributes {depth = 16 : i64, maskGran = 8 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, readUnderWrite = 0 : i32, width = 32 : ui32, writeClockIDs = [0 : i32], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32, initFilename = "", initIsBinary = false, initIsInline = false}
  hw.module @memTestFoo(%clock: i1, %rAddr: i4, %rEn: i1, %wAddr: i4, %wEn: i1, %wMask: i4, %wData: i32) -> (rData: i32) attributes {firrtl.moduleHierarchyFile = #hw.output_file<"testharness_hier.json", excludeFromFileList>} {
    %memory.R0_data = hw.instance "memory" @FIRRTLMem_1_1_0_32_16_1_1_0_1_a(R0_addr: %rAddr: i4, R0_en: %rEn: i1, R0_clk: %clock: i1, W0_addr: %wAddr: i4, W0_en: %wEn: i1, W0_clk: %clock: i1, W0_data: %wData: i32, W0_mask: %wMask: i4) -> (R0_data: i32)
    hw.output %memory.R0_data : i32
  }
  // COMMON-LABEL: hw.module @FIRRTLMem_1_1_0_32_16_1_1_0_1_a
  // CHECK-SAME: (%R0_addr: i4, %R0_en: i1, %R0_clk: i1, %W0_addr: i4, %W0_en: i1, %W0_clk: i1, %W0_data: i32, %W0_mask: i4) -> (R0_data: i32)
  // CHECK-NEXT:   %[[Memory:.+]] = sv.reg
  // VIVADO-SAME:  #sv.attribute<"rw_addr_collision" = "\22yes\22">
  // CHECK-SAME: !hw.inout<uarray<16xi32>>
  // CHECK:        %[[slot:.+]] = sv.array_index_inout %[[Memory]][%[[v3:.+]]] :
  // PRAMGAS: sv.attributes
  // CHECK-NEXT:   %[[v12:.+]] = sv.read_inout %[[slot]]
  // CHECK-NEXT:   %[[x_i32:.+]] = sv.constantX : i32
  // CHECK-NEXT:   %[[v13:.+]] = comb.mux %[[v1:.+]], %[[v12]], %[[x_i32]] : i32
  // CHECK-NEXT:   %[[v14:.+]] = comb.extract %W0_mask from 0 : (i4) -> i1
  // CHECK-NEXT:   %[[v15:.+]] = comb.extract %W0_data from 0 : (i32) -> i8
  // CHECK-NEXT:   %[[v16:.+]] = comb.extract %W0_mask from 1 : (i4) -> i1
  // CHECK-NEXT:   %[[v17:.+]] = comb.extract %W0_data from 8 : (i32) -> i8
  // CHECK-NEXT:   %[[v18:.+]] = comb.extract %W0_mask from 2 : (i4) -> i1
  // CHECK-NEXT:   %[[v19:.+]] = comb.extract %W0_data from 16 : (i32) -> i8
  // CHECK-NEXT:   %[[v20:.+]] = comb.extract %W0_mask from 3 : (i4) -> i1
  // CHECK-NEXT:   %[[v21:.+]] = comb.extract %W0_data from 24 : (i32) -> i8
  // CHECK-NEXT:   sv.always posedge %W0_clk {
  // CHECK-NEXT:     %[[v22:.+]] = comb.and %W0_en, %[[v14]] : i1
  // CHECK-NEXT:     sv.if %[[v22]]  {
  // CHECK-NEXT:       %[[v26:.+]] = sv.array_index_inout %[[Memory]][%W0_addr] : !hw.inout<uarray<16xi32>>, i4
  // CHECK-NEXT:      %[[c0_i32:.+]] = hw.constant 0 : i32
  // CHECK-NEXT:      %[[v220:.+]] = sv.indexed_part_select_inout %[[v26]][%[[c0_i32]] : 8] : !hw.inout<i32>, i32
  // CHECK-NEXT:      sv.passign %[[v220]], %[[v15]] : i8

  // IGNORE:      %[[Memory:.+]] = sv.reg : !hw.inout<uarray<16xi32>>
  // IGNORE-NEXT: %[[read_addr_inout:.+]] = sv.reg {{.*}} : !hw.inout<i4>
  // IGNORE-NEXT: sv.always posedge %R0_clk {
  // IGNORE-NEXT:   sv.if %R0_en {
  // IGNORE-NEXT:     sv.passign %[[read_addr_inout]], %R0_addr
  // IGNORE-NEXT:   }
  // IGNORE-NEXT: }
  // IGNORE-NEXT: %[[read_addr:.+]] = sv.read_inout %[[read_addr_inout]]
  // IGNORE-NEXT: %[[slot_read:.+]] = sv.array_index_inout %Memory[%[[read_addr]]]
  // IGNORE-NEXT: %[[result_read:.+]] = sv.read_inout %[[slot_read]]
  // IGNORE:      hw.output %[[result_read]]

  hw.module.generated @FIRRTLMem_1_1_0_32_16_1_1_0_1_b, @FIRRTLMem(%R0_addr: i4, %R0_en: i1, %R0_clk: i1, %W0_addr: i4, %W0_en: i1, %W0_clk: i1, %W0_data: i32, %W0_mask: i2) -> (R0_data: i32) attributes {depth = 16 : i64, maskGran = 16 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 2 : ui32, readUnderWrite = 0 : i32, width = 32 : ui32, writeClockIDs = [0 : i32], writeLatency = 3 : ui32, writeUnderWrite = 1 : i32, initFilename = "", initIsBinary = false, initIsInline = false}
  hw.module @memTestBar(%clock: i1, %rAddr: i4, %rEn: i1, %wAddr: i4, %wEn: i1, %wMask: i2, %wData: i32) -> (rData: i32) attributes {firrtl.moduleHierarchyFile = #hw.output_file<"testharness_hier.json", excludeFromFileList>} {
    %memory.R0_data = hw.instance "memory" @FIRRTLMem_1_1_0_32_16_1_1_0_1_b(R0_addr: %rAddr: i4, R0_en: %rEn: i1,
    R0_clk: %clock: i1, W0_addr: %wAddr: i4, W0_en: %wEn: i1, W0_clk: %clock: i1, W0_data: %wData: i32, W0_mask: %wMask:  i2) -> (R0_data: i32)
    hw.output %memory.R0_data : i32
  }
  // COMMON-LABEL:hw.module @FIRRTLMem_1_1_0_32_16_1_1_0_1_b
  // CHECK-SAME: (%R0_addr: i4, %R0_en: i1, %R0_clk: i1, %W0_addr: i4, %W0_en: i1, %W0_clk: i1, %W0_data: i32, %W0_mask: i2) -> (R0_data: i32)
  // CHECK:  %[[Memory0:.+]] = sv.reg : !hw.inout<uarray<16xi32>>
  // CHECK:  %[[v8:.+]] = sv.array_index_inout %[[Memory0]][%[[v7:.+]]] : !hw.inout<uarray<16xi32>>, i4
  // CHECK:  %[[v9:.+]] = sv.read_inout
  // CHECK:  %[[x_i32:.+]] = sv.constantX : i32
  // CHECK:  %[[v13:.+]] = comb.mux
  // CHECK:  %[[v24:.+]] = sv.reg {{.+}} : !hw.inout<i32>
  // CHECK:  sv.always posedge %W0_clk {
  // CHECK:    sv.passign %[[v22:.+]], %W0_data : i32
  // CHECK:    %[[v23:.+]] = sv.read_inout %[[v22]] : !hw.inout<i32>
  // CHECK:    sv.passign %[[v24:.+]], %[[v23]] : i32
  // CHECK:    sv.passign %[[v26:.+]], %W0_mask : i2
  // CHECK:    sv.passign %[[v28:.+]], %[[v27:.+]] : i2
  // CHECK:  }
  // CHECK:  %[[v25:.+]] = sv.read_inout %[[v24]] : !hw.inout<i32>
  // CHECK:  %[[v29:.+]] = sv.read_inout %[[v28]] : !hw.inout<i2>
  // CHECK:  %[[v30:.+]] = comb.extract %[[v29]] from 0 : (i2) -> i1
  // CHECK:  %[[v31:.+]] = comb.extract %[[v25]] from 0 : (i32) -> i16
  // CHECK:  %[[v32:.+]] = comb.extract %[[v29]] from 1 : (i2) -> i1
  // CHECK:  %[[v33:.+]] = comb.extract %[[v25]] from 16 : (i32) -> i16
  // CHECK:  sv.always posedge %W0_clk {
  // CHECK:    %[[v34:.+]] = comb.and %[[v21:.+]], %[[v30]] : i1
  // CHECK:    sv.if %[[v34]]  {
  // CHECK:      %[[v36:.+]] = sv.array_index_inout %[[Memory0]][%[[v117:.+]]] :
  // CHECK:      %c0_i32 = hw.constant 0 : i32
  // CHECK:      %[[v37:.+]] = sv.indexed_part_select_inout %[[v36]][%c0_i32 : 16] : !hw.inout<i32>, i32
  // CHECK:      sv.passign %[[v37]], %[[v31]] : i16
  // CHECK:    }
  // CHECK:  hw.output %[[v13]] : i32
  // CHECK:}

hw.module.generated @FIRRTLMem_1_1_1_16_10_2_4_0_0_multi, @FIRRTLMem(%ro_addr_0: i4, %ro_en_0: i1, %ro_clock_0: i1,%rw_addr_0:
i4, %rw_en_0: i1,  %rw_clock_0: i1, %rw_wmode_0: i1, %rw_wdata_0: i16,  %rw_wmask_0: i2,  %wo_addr_0: i4, %wo_en_0: i1,
%wo_clock_0: i1, %wo_data_0: i16, %wo_mask_0: i2) -> (ro_data_0: i16, rw_rdata_0: i16) attributes {depth = 10 : i64,
numReadPorts = 1 : ui32, numReadWritePorts = 1 : ui32,maskGran = 8 :ui32, numWritePorts = 1 : ui32, readLatency = 2 : ui32, readUnderWrite = 0 : i32, width = 16 : ui32, writeClockIDs = [], writeLatency = 4 : ui32, writeUnderWrite = 0 : i32, initFilename = "", initIsBinary = false, initIsInline = false}

// COMMON-LABEL:  hw.module @FIRRTLMem_1_1_1_16_10_2_4_0_0_multi
// CHECK-SAME: %ro_addr_0: i4, %ro_en_0: i1, %ro_clock_0: i1, %rw_addr_0: i4,
// CHECK-SAME: %rw_en_0: i1, %rw_clock_0: i1, %rw_wmode_0: i1, %rw_wdata_0: i16,
// CHECK-SAME: %rw_wmask_0: i2, %wo_addr_0: i4, %wo_en_0: i1, %wo_clock_0: i1,
// CHECK-SAME: %wo_data_0: i16, %wo_mask_0: i2
// CHECK:    %[[Memory0:.+]] = sv.reg : !hw.inout<uarray<10xi16>>
// CHECK:    %[[v8:.+]] = sv.array_index_inout %[[Memory0]][%[[v7:.+]]] :
// CHECK:    %[[v12:.+]] = sv.read_inout %[[v8]]
// CHECK:    sv.always posedge %rw_clock_0 {
// CHECK:      sv.passign %[[v38:.+]], %rw_wmask_0 : i2
// CHECK:    %[[v44:.+]] = comb.extract %[[v43:.+]] from 0 : (i2) -> i1
// CHECK:    %[[v45:.+]] = comb.extract %[[v37:.+]] from 0 : (i16) -> i8
// CHECK:    %[[v46:.+]] = comb.extract %[[v43]] from 1 : (i2) -> i1
// CHECK:    %[[v47:.+]] = comb.extract %[[v37]] from 8 : (i16) -> i8
// CHECK:    %[[v52:.+]] = sv.array_index_inout %[[Memory0]][%[[v19:.+]]] :
// CHECK:    %[[v56:.+]] = sv.read_inout %[[v52]]
// CHECK:    sv.always posedge %wo_clock_0 {
// CHECK:      sv.passign %[[v70:.+]], %wo_data_0 : i16
// CHECK:      sv.passign %[[v76:.+]], %wo_mask_0 : i2
// CHECK:    }
// CHECK:    %[[v82:.+]] = comb.extract %[[v81:.+]] from 0 : (i2) -> i1
// CHECK:    %[[v83:.+]] = comb.extract %[[v75:.+]] from 0 : (i16) -> i8
// CHECK:    %[[v84:.+]] = comb.extract %[[v81]] from 1 : (i2) -> i1
// CHECK:    %[[v85:.+]] = comb.extract %[[v75]] from 8 : (i16) -> i8
// CHECK:    sv.always posedge %wo_clock_0 {
// CHECK:      %[[v86:.+]] = comb.and %[[v69:.+]], %[[v82]] : i1
// CHECK:      sv.if %[[v86]]  {
// CHECK:        %[[v88:.+]] = sv.array_index_inout %[[Memory0]][%[[v63:.+]]] :
// CHECK:        %[[vv83:.+]] = sv.indexed_part_select_inout %[[v88]][%[[c0_i32:.+]] : 8] : !hw.inout<i16>, i32
// CHECK:        sv.passign %[[vv83]], %[[v83]] : i8
// CHECK:      }
// CHECK:      sv.if %[[v87:.+]]  {
// CHECK:        %[[v88:.+]] = sv.array_index_inout %[[Memory0]][%[[v63]]] : !hw.inout<uarray<10xi16>>, i4
// CHECK:        %[[c8_i32:.+]] = hw.constant 8 : i32
// CHECK:        %[[vv83:.+]] = sv.indexed_part_select_inout %[[v88]][%[[c8_i32]] : 8] : !hw.inout<i16>, i32
// CHECK:        sv.passign %[[vv83]], %[[v85]] : i8
// CHECK:      }
// CHECK:    }

// Ensure state is cleaned up between the expansion of modules.
// See https://github.com/llvm/circt/pull/2769

// COMMON-LABEL: hw.module @PR2769
// CHECK-NOT: _GEN
hw.module.generated @PR2769, @FIRRTLMem(%ro_addr_0: i4, %ro_en_0: i1, %ro_clock_0: i1,%rw_addr_0: i4, %rw_en_0: i1,  %rw_clock_0: i1, %rw_wmode_0: i1, %rw_wdata_0: i16,  %wo_addr_0: i4, %wo_en_0: i1, %wo_clock_0: i1, %wo_data_0: i16) -> (ro_data_0: i16, rw_rdata_0: i16) attributes {depth = 10 : i64, numReadPorts = 1 : ui32, numReadWritePorts = 1 : ui32, numWritePorts = 1 : ui32, readLatency = 0 : ui32, readUnderWrite = 0 : i32, width = 16 : ui32, writeClockIDs = [], writeLatency = 1 : ui32, writeUnderWrite = 0 : i32, initFilename = "", initIsBinary = false, initIsInline = false}

// COMMON-LABEL: hw.module @RandomizeWeirdWidths
// CHECK: sv.ifdef.procedural "RANDOMIZE_MEM_INIT"
// CHECK: %[[INOUT:.+]] = sv.array_index_inout %Memory[%i]
// CHECK: %[[EXTRACT:.+]] = comb.extract %{{.+}} from 0 : (i160) -> i145
// CHECK-NEXT: sv.bpassign %[[INOUT]], %[[EXTRACT]] : i145
hw.module.generated @RandomizeWeirdWidths, @FIRRTLMem(%ro_addr_0: i4, %ro_en_0: i1, %ro_clock_0: i1,%rw_addr_0: i4, %rw_en_0: i1,  %rw_clock_0: i1, %rw_wmode_0: i1, %rw_wdata_0: i145, %wo_addr_0: i4, %wo_en_0: i1, %wo_clock_0: i1, %wo_data_0: i145) -> (ro_data_0: i145, rw_rdata_0: i145) attributes {depth = 10 : i64, numReadPorts = 1 : ui32, numReadWritePorts = 1 : ui32, numWritePorts = 1 : ui32, readLatency = 2 : ui32, readUnderWrite = 0 : i32, width = 145 : ui32, writeClockIDs = [], writeLatency = 4 : ui32, writeUnderWrite = 0 : i32, initFilename = "", initIsBinary = false, initIsInline = false}

// COMMON-LABEL: hw.module @ReadWriteWithHighReadLatency
hw.module.generated @ReadWriteWithHighReadLatency, @FIRRTLMem(%rw_addr: i4, %rw_en: i1,  %rw_clock: i1, %rw_wmode: i1, %rw_wdata: i16) -> (rw_rdata: i16) attributes {depth = 16 : i64, numReadPorts = 0 : ui32, numReadWritePorts = 1 : ui32, numWritePorts = 0 : ui32, readLatency = 4 : ui32, readUnderWrite = 0 : i32, width = 16 : ui32, writeClockIDs = [], writeLatency = 3 : ui32, writeUnderWrite = 0 : i32, initFilename = "", initIsBinary = false, initIsInline = false}
// CHECK: [[MEM:%.+]] = sv.reg

// Common pipeline stages (2x)
// CHECK: sv.passign [[ADDR_0:%.+]], %rw_addr
// CHECK: [[ADDR_0R:%.+]] = sv.read_inout [[ADDR_0]]
// CHECK: sv.passign [[ADDR_1:%.+]], [[ADDR_0R]]

// CHECK: sv.passign [[EN_0:%.+]], %rw_en
// CHECK: [[EN_0R:%.+]] = sv.read_inout [[EN_0]]
// CHECK: sv.passign [[EN_1:%.+]], [[EN_0R]]

// CHECK: sv.passign [[WMODE_0:%.+]], %rw_wmode
// CHECK: [[WMODE_0R:%.+]] = sv.read_inout [[WMODE_0]]
// CHECK: sv.passign [[WMODE_1:%.+]], [[WMODE_0R]]

// CHECK: [[ADDR_1R:%.+]] = sv.read_inout [[ADDR_1]]
// CHECK: [[EN_1R:%.+]] = sv.read_inout [[EN_1]]
// CHECK: [[WMODE_1R:%.+]] = sv.read_inout [[WMODE_1]]

// Additional read pipeline stages (2x)
// CHECK: sv.passign [[READ_ADDR_2:%.+]], [[ADDR_1R]]
// CHECK: [[READ_ADDR_2R:%.+]] = sv.read_inout [[READ_ADDR_2]]
// CHECK: sv.passign [[READ_ADDR_3:%.+]], [[READ_ADDR_2R]]

// CHECK: sv.passign [[READ_EN_2:%.+]], [[EN_1R]]
// CHECK: [[READ_EN_2R:%.+]] = sv.read_inout [[READ_EN_2]]
// CHECK: sv.passign [[READ_EN_3:%.+]], [[READ_EN_2R]]

// CHECK: sv.passign [[READ_WMODE_2:%.+]], [[WMODE_1R]]
// CHECK: [[READ_WMODE_2R:%.+]] = sv.read_inout [[READ_WMODE_2]]
// CHECK: sv.passign [[READ_WMODE_3:%.+]], [[READ_WMODE_2R]]

// CHECK: [[READ_ADDR_3R:%.+]] = sv.read_inout [[READ_ADDR_3]]
// CHECK: [[READ_EN_3R:%.+]] = sv.read_inout [[READ_EN_3]]
// CHECK: [[READ_WMODE_3R:%.+]] = sv.read_inout [[READ_WMODE_3]]

// Read port
// CHECK: [[RMODE:%.+]] = comb.icmp eq [[READ_WMODE_3R]], %false
// CHECK: [[RCOND:%.+]] = comb.and [[READ_EN_3R]], [[RMODE]]
// CHECK: [[RPTR:%.+]] = sv.array_index_inout [[MEM]][[[READ_ADDR_3R]]]

// Write port
// CHECK: sv.always
// CHECK: [[TMP:%.+]] = comb.and [[WMODE_1R]], %true
// CHECK: [[WCOND:%.+]] comb.and [[EN_1R]], [[TMP]]
// CHECK: [[WPTR:%.+]] = sv.array_index_inout [[MEM]][[[ADDR_1R]]]

// COMMON-LABEL: hw.module @ReadWriteWithHighWriteLatency
hw.module.generated @ReadWriteWithHighWriteLatency, @FIRRTLMem(%rw_addr: i4, %rw_en: i1,  %rw_clock: i1, %rw_wmode: i1, %rw_wdata: i16) -> (rw_rdata: i16) attributes {depth = 16 : i64, numReadPorts = 0 : ui32, numReadWritePorts = 1 : ui32, numWritePorts = 0 : ui32, readLatency = 2 : ui32, readUnderWrite = 0 : i32, width = 16 : ui32, writeClockIDs = [], writeLatency = 5 : ui32, writeUnderWrite = 0 : i32, initFilename = "", initIsBinary = false, initIsInline = false}
// CHECK: [[MEM:%.+]] = sv.reg

// Common pipeline stages (2x)
// CHECK: sv.passign [[ADDR_0:%.+]], %rw_addr
// CHECK: [[ADDR_0R:%.+]] = sv.read_inout [[ADDR_0]]
// CHECK: sv.passign [[ADDR_1:%.+]], [[ADDR_0R]]

// CHECK: sv.passign [[EN_0:%.+]], %rw_en
// CHECK: [[EN_0R:%.+]] = sv.read_inout [[EN_0]]
// CHECK: sv.passign [[EN_1:%.+]], [[EN_0R]]

// CHECK: sv.passign [[WMODE_0:%.+]], %rw_wmode
// CHECK: [[WMODE_0R:%.+]] = sv.read_inout [[WMODE_0]]
// CHECK: sv.passign [[WMODE_1:%.+]], [[WMODE_0R]]

// CHECK: [[ADDR_1R:%.+]] = sv.read_inout [[ADDR_1]]
// CHECK: [[EN_1R:%.+]] = sv.read_inout [[EN_1]]
// CHECK: [[WMODE_1R:%.+]] = sv.read_inout [[WMODE_1]]

// Additional write pipeline stages (2x)
// CHECK: sv.passign [[WRITE_ADDR_2:%.+]], [[ADDR_1R]]
// CHECK: [[WRITE_ADDR_2R:%.+]] = sv.read_inout [[WRITE_ADDR_2]]
// CHECK: sv.passign [[WRITE_ADDR_3:%.+]], [[WRITE_ADDR_2R]]

// CHECK: sv.passign [[WRITE_EN_2:%.+]], [[EN_1R]]
// CHECK: [[WRITE_EN_2R:%.+]] = sv.read_inout [[WRITE_EN_2]]
// CHECK: sv.passign [[WRITE_EN_3:%.+]], [[WRITE_EN_2R]]

// CHECK: sv.passign [[WRITE_WMODE_2:%.+]], [[WMODE_1R]]
// CHECK: [[WRITE_WMODE_2R:%.+]] = sv.read_inout [[WRITE_WMODE_2]]
// CHECK: sv.passign [[WRITE_WMODE_3:%.+]], [[WRITE_WMODE_2R]]

// CHECK: [[WRITE_ADDR_3R:%.+]] = sv.read_inout [[WRITE_ADDR_3]]
// CHECK: [[WRITE_EN_3R:%.+]] = sv.read_inout [[WRITE_EN_3]]
// CHECK: [[WRITE_WMODE_3R:%.+]] = sv.read_inout [[WRITE_WMODE_3]]

// Read port
// CHECK: [[RMODE:%.+]] = comb.icmp eq [[WMODE_1R]], %false
// CHECK: [[RCOND:%.+]] = comb.and [[EN_1R]], [[RMODE]]
// CHECK: [[RPTR:%.+]] = sv.array_index_inout [[MEM]][[[ADDR_1R]]]

// Write port
// CHECK: sv.always
// CHECK: [[TMP:%.+]] = comb.and [[WRITE_WMODE_3R]], %true
// CHECK: [[WCOND:%.+]] comb.and [[WRITE_EN_3R]], [[TMP]]
// CHECK: [[WPTR:%.+]] = sv.array_index_inout [[MEM]][[[WRITE_ADDR_3R]]]

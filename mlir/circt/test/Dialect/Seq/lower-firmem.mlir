// RUN: circt-opt --lower-seq-firmem %s --verify-diagnostics | FileCheck %s

// CHECK: hw.generator.schema @FIRRTLMem, "FIRRTL_Memory", ["depth", "numReadPorts", "numWritePorts", "numReadWritePorts", "readLatency", "writeLatency", "width", "maskGran", "readUnderWrite", "writeUnderWrite", "writeClockIDs", "initFilename", "initIsBinary", "initIsInline"]

// CHECK: sv.macro.decl @SomeMacro
sv.macro.decl @SomeMacro

// CHECK: hw.module.generated @m0_mem1_12x42, @FIRRTLMem(%R0_addr: i4, %R0_en: i1, %R0_clk: i1) -> (R0_data: i42) attributes {depth = 12 : i64, initFilename = "", initIsBinary = false, initIsInline = false, maskGran = 42 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 0 : ui32, readLatency = 0 : ui32, readUnderWrite = 0 : i32, width = 42 : ui32, writeClockIDs = [], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}
// CHECK: hw.module.generated @m0_mem2_12x42, @FIRRTLMem(%W0_addr: i4, %W0_en: i1, %W0_clk: i1, %W0_data: i42, %W0_mask: i2) attributes {depth = 12 : i64, initFilename = "", initIsBinary = false, initIsInline = false, maskGran = 21 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 0 : ui32, readUnderWrite = 0 : i32, width = 42 : ui32, writeClockIDs = [0 : i32], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}
// CHECK: hw.module.generated @m0_mem3_12x42, @FIRRTLMem(%RW0_addr: i4, %RW0_en: i1, %RW0_clk: i1, %RW0_wmode: i1, %RW0_wdata: i42, %RW0_wmask: i3) -> (RW0_rdata: i42) attributes {depth = 12 : i64, initFilename = "", initIsBinary = false, initIsInline = false, maskGran = 14 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 1 : ui32, numWritePorts = 0 : ui32, readLatency = 0 : ui32, readUnderWrite = 0 : i32, width = 42 : ui32, writeClockIDs = [0 : i32], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}
// CHECK: hw.module.generated @m0_mem4_12x42, @FIRRTLMem(%R0_addr: i4, %R0_en: i1, %R0_clk: i1, %RW0_addr: i4, %RW0_en: i1, %RW0_clk: i1, %RW0_wmode: i1, %RW0_wdata: i42, %RW0_wmask: i6, %W0_addr: i4, %W0_en: i1, %W0_clk: i1, %W0_data: i42, %W0_mask: i6) -> (R0_data: i42, RW0_rdata: i42) attributes {depth = 12 : i64, initFilename = "", initIsBinary = false, initIsInline = false, maskGran = 7 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 1 : ui32, numWritePorts = 1 : ui32, readLatency = 0 : ui32, readUnderWrite = 0 : i32, width = 42 : ui32, writeClockIDs = [0 : i32, 0 : i32], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}

// CHECK-LABEL: hw.module @Foo
hw.module @Foo(%clk: i1, %en: i1, %addr: i4, %wdata: i42, %wmode: i1, %mask2: i2, %mask3: i3, %mask6: i6) {
  // CHECK-NEXT: [[TMP0:%.+]] = hw.instance "m0_mem1A_ext" @m0_mem1_12x42(R0_addr: %addr: i4, R0_en: %en: i1, R0_clk: %clk: i1) -> (R0_data: i42)
  // CHECK-NEXT: [[TMP1:%.+]] = hw.instance "m0_mem1B_ext" @m0_mem1_12x42(R0_addr: %addr: i4, R0_en: %en: i1, R0_clk: %clk: i1) -> (R0_data: i42)
  // CHECK-NEXT: comb.xor [[TMP0]], [[TMP1]]
  %m0_mem1A = seq.firmem 0, 1, undefined, port_order : <12 x 42>
  %m0_mem1B = seq.firmem 0, 1, undefined, port_order : <12 x 42>
  %0 = seq.firmem.read_port %m0_mem1A[%addr], clock %clk enable %en : <12 x 42>
  %1 = seq.firmem.read_port %m0_mem1B[%addr], clock %clk enable %en : <12 x 42>
  comb.xor %0, %1 : i42

  // CHECK-NEXT: hw.instance "m0_mem2_ext" @m0_mem2_12x42(W0_addr: %addr: i4, W0_en: %en: i1, W0_clk: %clk: i1, W0_data: %wdata: i42, W0_mask: %mask2: i2) -> ()
  %m0_mem2 = seq.firmem 0, 1, undefined, port_order : <12 x 42, mask 2>
  seq.firmem.write_port %m0_mem2[%addr] = %wdata, clock %clk enable %en mask %mask2 : <12 x 42, mask 2>, i2

  // CHECK-NEXT: [[TMP:%.+]] = hw.instance "m0_mem3_ext" @m0_mem3_12x42(RW0_addr: %addr: i4, RW0_en: %en: i1, RW0_clk: %clk: i1, RW0_wmode: %wmode: i1, RW0_wdata: %wdata: i42, RW0_wmask: %mask3: i3) -> (RW0_rdata: i42)
  // CHECK-NEXT: comb.xor [[TMP]]
  %m0_mem3 = seq.firmem 0, 1, undefined, port_order : <12 x 42, mask 3>
  %2 = seq.firmem.read_write_port %m0_mem3[%addr] = %wdata if %wmode, clock %clk enable %en mask %mask3 : <12 x 42, mask 3>, i3
  comb.xor %2 : i42

  // CHECK-NEXT: [[TMP0:%.+]], [[TMP1:%.+]] = hw.instance "m0_mem4_ext" @m0_mem4_12x42(R0_addr: %addr: i4, R0_en: %en: i1, R0_clk: %clk: i1, RW0_addr: %addr: i4, RW0_en: %en: i1, RW0_clk: %clk: i1, RW0_wmode: %wmode: i1, RW0_wdata: %wdata: i42, RW0_wmask: %mask6: i6, W0_addr: %addr: i4, W0_en: %en: i1, W0_clk: %clk: i1, W0_data: %wdata: i42, W0_mask: %mask6: i6) -> (R0_data: i42, RW0_rdata: i42)
  // CHECK-NEXT: comb.xor [[TMP0]], [[TMP1]]
  %m0_mem4 = seq.firmem 0, 1, undefined, port_order : <12 x 42, mask 6>
  %3 = seq.firmem.read_port %m0_mem4[%addr], clock %clk enable %en : <12 x 42, mask 6>
  %4 = seq.firmem.read_write_port %m0_mem4[%addr] = %wdata if %wmode, clock %clk enable %en mask %mask6 : <12 x 42, mask 6>, i6
  seq.firmem.write_port %m0_mem4[%addr] = %wdata, clock %clk enable %en mask %mask6 : <12 x 42, mask 6>, i6
  comb.xor %3, %4 : i42
}

// CHECK: hw.module.generated @m1_mem1_24x1337, @FIRRTLMem() attributes {depth = 24 : i64, initFilename = "", initIsBinary = false, initIsInline = false, maskGran = 1337 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 0 : ui32, output_file = "foo", readLatency = 0 : ui32, readUnderWrite = 0 : i32, width = 1337 : ui32, writeClockIDs = [], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}
// CHECK: hw.module.generated @m1_mem2_24x1337, @FIRRTLMem() attributes {depth = 24 : i64, initFilename = "", initIsBinary = false, initIsInline = false, maskGran = 1337 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 0 : ui32, output_file = "bar", readLatency = 0 : ui32, readUnderWrite = 0 : i32, width = 1337 : ui32, writeClockIDs = [], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}

// CHECK-LABEL: hw.module @SeparateOutputFiles
hw.module @SeparateOutputFiles() {
  // CHECK-NEXT: hw.instance "m1_mem1_ext" @m1_mem1_24x1337(
  // CHECK-NEXT: hw.instance "m1_mem2_ext" @m1_mem2_24x1337(
  %m1_mem1 = seq.firmem 0, 1, undefined, port_order {output_file = "foo"} : <24 x 1337>
  %m1_mem2 = seq.firmem 0, 1, undefined, port_order {output_file = "bar"} : <24 x 1337>
}

// CHECK: hw.module.generated @foo_m2_mem1_24x9001, @FIRRTLMem() attributes {depth = 24 : i64, initFilename = "", initIsBinary = false, initIsInline = false, maskGran = 9001 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 0 : ui32, readLatency = 0 : ui32, readUnderWrite = 0 : i32, width = 9001 : ui32, writeClockIDs = [], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}
// CHECK: hw.module.generated @bar_m2_mem2_24x9001, @FIRRTLMem() attributes {depth = 24 : i64, initFilename = "", initIsBinary = false, initIsInline = false, maskGran = 9001 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 0 : ui32, readLatency = 0 : ui32, readUnderWrite = 0 : i32, width = 9001 : ui32, writeClockIDs = [], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}
// CHECK: hw.module.generated @uwu_m2_mem_24x9001, @FIRRTLMem() attributes {depth = 24 : i64, initFilename = "", initIsBinary = false, initIsInline = false, maskGran = 9001 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 0 : ui32, readLatency = 0 : ui32, readUnderWrite = 0 : i32, width = 9001 : ui32, writeClockIDs = [], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}

// CHECK-LABEL: hw.module @SeparatePrefices
hw.module @SeparatePrefices() {
  // CHECK-NEXT: hw.instance "m2_mem1_ext" @foo_m2_mem1_24x9001(
  %m2_mem1 = seq.firmem 0, 1, undefined, port_order {prefix = "foo_"} : <24 x 9001>
  // CHECK-NEXT: hw.instance "m2_mem2_ext" @bar_m2_mem2_24x9001(
  %m2_mem2 = seq.firmem 0, 1, undefined, port_order {prefix = "bar_"} : <24 x 9001>
  // CHECK-NEXT: hw.instance "m2_mem3_ext" @uwu_m2_mem_24x9001(
  // CHECK-NEXT: hw.instance "m2_mem4_ext" @uwu_m2_mem_24x9001(
  %m2_mem3 = seq.firmem 0, 1, undefined, port_order {prefix = "uwu_"} : <24 x 9001>
  %m2_mem4 = seq.firmem 0, 1, undefined, port_order {prefix = "uwu_"} : <24 x 9001>
}

// CHECK: hw.module.generated @m3_mem_12x8, @FIRRTLMem(%W0_addr: i4, %W0_en: i1, %W0_clk: i1, %W0_data: i8, %W1_addr: i4, %W1_en: i1, %W1_clk: i1, %W1_data: i8) attributes {depth = 12 : i64, initFilename = "", initIsBinary = false, initIsInline = false, maskGran = 8 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 2 : ui32, readLatency = 0 : ui32, readUnderWrite = 0 : i32, width = 8 : ui32, writeClockIDs = [0 : i32, 0 : i32], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}
// CHECK: hw.module.generated @m3_mem2_12x8, @FIRRTLMem(%W0_addr: i4, %W0_en: i1, %W0_clk: i1, %W0_data: i8, %W1_addr: i4, %W1_en: i1, %W1_clk: i1, %W1_data: i8) attributes {depth = 12 : i64, initFilename = "", initIsBinary = false, initIsInline = false, maskGran = 8 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 2 : ui32, readLatency = 0 : ui32, readUnderWrite = 0 : i32, width = 8 : ui32, writeClockIDs = [0 : i32, 1 : i32], writeLatency = 1 : ui32, writeUnderWrite = 1 : i32}

// CHECK-LABEL: hw.module @MemoryWritePortBehavior
hw.module @MemoryWritePortBehavior(%clock1: i1, %clock2: i1) {
  %z_i8 = sv.constantZ : i8
  %z_i1 = sv.constantZ : i1
  %z_i4 = sv.constantZ : i4

  // This memory has both write ports driven by the same clock.  It should be
  // lowered to an "aa" memory. Even if the clock is passed via different
  // wires, we should identify the clocks to be same.
  //
  // CHECK: hw.instance "m3_mem1_ext" @m3_mem_12x8
  %m3_mem1 = seq.firmem 0, 1, undefined, port_order : <12 x 8>
  %cwire1 = hw.wire %clock1 : i1
  %cwire2 = hw.wire %clock1 : i1
  seq.firmem.write_port %m3_mem1[%z_i4] = %z_i8, clock %cwire1 enable %z_i1 : <12 x 8>
  seq.firmem.write_port %m3_mem1[%z_i4] = %z_i8, clock %cwire2 enable %z_i1 : <12 x 8>

  // This memory has different clocks for each write port. It should be
  // lowered to an "ab" memory.
  //
  // CHECK: hw.instance "m3_mem2_ext" @m3_mem2_12x8
  %m3_mem2 = seq.firmem 0, 1, undefined, port_order : <12 x 8>
  seq.firmem.write_port %m3_mem2[%z_i4] = %z_i8, clock %clock1 enable %z_i1 : <12 x 8>
  seq.firmem.write_port %m3_mem2[%z_i4] = %z_i8, clock %clock2 enable %z_i1 : <12 x 8>

  // This memory is the same as the first memory, but a node is used to alias
  // the second write port clock (e.g., this could be due to a dont touch
  // annotation blocking this from being optimized away). This should be
  // lowered to an "aa" since they are identical.
  //
  // CHECK: hw.instance "m3_mem3_ext" @m3_mem_12x8
  %m3_mem3 = seq.firmem 0, 1, undefined, port_order : <12 x 8>
  seq.firmem.write_port %m3_mem3[%z_i4] = %z_i8, clock %clock1 enable %z_i1 : <12 x 8>
  seq.firmem.write_port %m3_mem3[%z_i4] = %z_i8, clock %clock1 enable %z_i1 : <12 x 8>
}

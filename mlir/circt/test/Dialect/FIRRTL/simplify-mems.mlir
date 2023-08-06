// RUN: circt-opt --split-input-file -canonicalize='top-down=true region-simplify=true' %s | FileCheck %s


firrtl.circuit "ReadOnlyMemory" {
  // CHECK-LABEL: firrtl.module public @ReadOnlyMemory
  firrtl.module public @ReadOnlyMemory(in %clock: !firrtl.clock, in %addr: !firrtl.uint<4>) {

    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %Memory_r = firrtl.mem Undefined
      {
        depth = 12 : i64,
        name = "Memory",
        portNames = ["read"],
        readLatency = 0 : i32,
        writeLatency = 1 : i32
      } : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>


    // CHECK-NOT: firrtl.mem
    %2 = firrtl.subfield %Memory_r[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    firrtl.connect %2, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %3 = firrtl.subfield %Memory_r[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    firrtl.connect %3, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %4 = firrtl.subfield %Memory_r[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    firrtl.connect %4, %clock : !firrtl.clock, !firrtl.clock
  }
}

// -----

firrtl.circuit "WriteOnlyMemory" {
  // CHECK-LABEL: firrtl.module public @WriteOnlyMemory
  firrtl.module public @WriteOnlyMemory(in %clock: !firrtl.clock, in %addr: !firrtl.uint<4>, in %indata: !firrtl.uint<42>) {

    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %Memory_write = firrtl.mem Undefined
      {
        depth = 12 : i64,
        name = "Memory",
        portNames = ["write"],
        readLatency = 0 : i32,
        writeLatency = 1 : i32
      } : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>

    // CHECK-NOT: firrtl.mem
    %10 = firrtl.subfield %Memory_write[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    firrtl.connect %10, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %11 = firrtl.subfield %Memory_write[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    firrtl.connect %11, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %12 = firrtl.subfield %Memory_write[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    firrtl.connect %12, %clock : !firrtl.clock, !firrtl.clock
    %13 = firrtl.subfield %Memory_write[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    firrtl.connect %13, %indata : !firrtl.uint<42>, !firrtl.uint<42>
    %14 = firrtl.subfield %Memory_write[mask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    firrtl.connect %14, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "ReadWriteToWrite" {
  firrtl.module public @ReadWriteToWrite(in %clock: !firrtl.clock, in %addr: !firrtl.uint<4>, in %indata: !firrtl.uint<42>, out %result: !firrtl.uint<42>) {

    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>

    // CHECK: %Memory_rw, %Memory_r = firrtl.mem  Undefined
    // CHECK-SAME: !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    // CHECK-SAME: !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>

    %Memory_rw, %Memory_r = firrtl.mem Undefined
      {
        depth = 12 : i64,
        name = "Memory",
        portNames = ["rw", "r"],
        readLatency = 0 : i32,
        writeLatency = 1 : i32
      } :
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>,
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>

    // CHECK: [[ADDR:%.+]] = firrtl.subfield %Memory_rw[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    // CHECK: [[END:%.+]] = firrtl.subfield %Memory_rw[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    // CHECK: [[CLK:%.+]] = firrtl.subfield %Memory_rw[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    // CHECK: [[DATA:%.+]] = firrtl.subfield %Memory_rw[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    // CHECK: [[MASK:%.+]] = firrtl.subfield %Memory_rw[mask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    // CHECK: [[DUMMY_WMODE:%.+]] = firrtl.wire : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect [[ADDR]], %addr : !firrtl.uint<4>
    // CHECK: firrtl.strictconnect [[END]], %c1_ui1 : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect [[CLK]], %clock : !firrtl.clock
    // CHECK: firrtl.strictconnect [[DUMMY_WMODE]], %c1_ui1 : !firrtl.uint<1>
    // CHECK: firrtl.strictconnect [[DATA]], %indata : !firrtl.uint<42>
    // CHECK: firrtl.strictconnect [[MASK]], %c1_ui1 : !firrtl.uint<1>

    %0 = firrtl.subfield %Memory_rw[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    firrtl.connect %0, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %1 = firrtl.subfield %Memory_rw[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    firrtl.connect %1, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %2 = firrtl.subfield %Memory_rw[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    firrtl.connect %2, %clock : !firrtl.clock, !firrtl.clock
    %3 = firrtl.subfield %Memory_rw[wmode] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    firrtl.connect %3, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %4 = firrtl.subfield %Memory_rw[wdata] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    firrtl.connect %4, %indata : !firrtl.uint<42>, !firrtl.uint<42>
    %5 = firrtl.subfield %Memory_rw[wmask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    firrtl.connect %5, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>

    %6 = firrtl.subfield %Memory_r[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    firrtl.connect %6, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %7 = firrtl.subfield %Memory_r[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    firrtl.connect %7, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %8 = firrtl.subfield %Memory_r[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    firrtl.connect %8, %clock : !firrtl.clock, !firrtl.clock
    %9 = firrtl.subfield %Memory_r[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    firrtl.connect %result, %9 : !firrtl.uint<42>, !firrtl.uint<42>
  }
}

// -----

firrtl.circuit "UnusedPorts" {
  firrtl.module public @UnusedPorts(
      in %clock: !firrtl.clock,
      in %addr: !firrtl.uint<4>,
      in %in_data: !firrtl.uint<42>,
      in %wmode_rw: !firrtl.uint<1>,
      out %result_read: !firrtl.uint<42>,
      out %result_rw: !firrtl.uint<42>,
      out %result_pinned: !firrtl.uint<42>) {

    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>

    // CHECK: %Memory_pinned = firrtl.mem  Undefined
    // CHECK-SAME: !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>

    %Memory_read, %Memory_rw, %Memory_write, %Memory_pinned = firrtl.mem Undefined
      {
        depth = 12 : i64,
        name = "Memory",
        portNames = ["read", "rw", "write", "pinned"],
        readLatency = 0 : i32,
        writeLatency = 1 : i32
      } :
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>,
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>,
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>,
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>

    // CHECK: [[REG1:%.+]] = firrtl.reg %c0_clock : !firrtl.clock, !firrtl.uint<42>
    // CHECK: [[REG2:%.+]] = firrtl.reg %c0_clock : !firrtl.clock, !firrtl.uint<42>
    // CHECK: firrtl.strictconnect %result_read, [[REG1]] : !firrtl.uint<42>
    %read_addr = firrtl.subfield %Memory_read[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    firrtl.connect %read_addr, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %read_en = firrtl.subfield %Memory_read[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    firrtl.connect %read_en, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %read_clk = firrtl.subfield %Memory_read[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    firrtl.connect %read_clk, %clock : !firrtl.clock, !firrtl.clock
    %read_data = firrtl.subfield %Memory_read[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    firrtl.connect %result_read, %read_data : !firrtl.uint<42>, !firrtl.uint<42>

    // CHECK: firrtl.strictconnect %result_rw, [[REG2]] : !firrtl.uint<42>
    %rw_addr = firrtl.subfield %Memory_rw[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    firrtl.connect %rw_addr, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %rw_en = firrtl.subfield %Memory_rw[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    firrtl.connect %rw_en, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %rw_clk = firrtl.subfield %Memory_rw[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    firrtl.connect %rw_clk, %clock : !firrtl.clock, !firrtl.clock
    %rw_rdata = firrtl.subfield %Memory_rw[rdata] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    firrtl.connect %result_rw, %rw_rdata : !firrtl.uint<42>, !firrtl.uint<42>
    %rw_wmode = firrtl.subfield %Memory_rw[wmode] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    firrtl.connect %rw_wmode, %wmode_rw : !firrtl.uint<1>, !firrtl.uint<1>
    %rw_wdata = firrtl.subfield %Memory_rw[wdata] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    firrtl.connect %rw_wdata, %in_data : !firrtl.uint<42>, !firrtl.uint<42>
    %rw_wmask = firrtl.subfield %Memory_rw[wmask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    firrtl.connect %rw_wmask, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>

    %write_addr = firrtl.subfield %Memory_write[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    firrtl.connect %write_addr, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %write_en = firrtl.subfield %Memory_write[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    firrtl.connect %write_en, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %write_clk = firrtl.subfield %Memory_write[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    firrtl.connect %write_clk, %clock : !firrtl.clock, !firrtl.clock
    %write_data = firrtl.subfield %Memory_write[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    firrtl.connect %write_data, %in_data : !firrtl.uint<42>, !firrtl.uint<42>
    %write_mask = firrtl.subfield %Memory_write[mask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    firrtl.connect %write_mask, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>

    %pinned_addr = firrtl.subfield %Memory_pinned[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    firrtl.connect %pinned_addr, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %pinned_en = firrtl.subfield %Memory_pinned[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    firrtl.connect %pinned_en, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %pinned_clk = firrtl.subfield %Memory_pinned[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    firrtl.connect %pinned_clk, %clock : !firrtl.clock, !firrtl.clock
    %pinned_rdata = firrtl.subfield %Memory_pinned[rdata] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    firrtl.connect %result_pinned, %pinned_rdata : !firrtl.uint<42>, !firrtl.uint<42>
    %pinned_wmode = firrtl.subfield %Memory_pinned[wmode] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    firrtl.connect %pinned_wmode, %wmode_rw : !firrtl.uint<1>, !firrtl.uint<1>
    %pinned_wdata = firrtl.subfield %Memory_pinned[wdata] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    firrtl.connect %pinned_wdata, %in_data : !firrtl.uint<42>, !firrtl.uint<42>
    %pinned_wmask = firrtl.subfield %Memory_pinned[wmask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    firrtl.connect %pinned_wmask, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "UnusedBits" {
  firrtl.module public @UnusedBits(
      in %clock: !firrtl.clock,
      in %addr: !firrtl.uint<4>,
      in %in_data: !firrtl.uint<42>,
      in %wmode_rw: !firrtl.uint<1>,
      out %result_read: !firrtl.uint<5>,
      out %result_rw: !firrtl.uint<5>) {

    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>

    // CHECK: %Memory_read, %Memory_rw, %Memory_write = firrtl.mem Undefined
    // CHECK-SAME: !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<10>>
    // CHECK-SAME: !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<10>, wmode: uint<1>, wdata: uint<10>, wmask: uint<1>>
    // CHECK-SAME: !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<10>, mask: uint<1>>
    %Memory_read, %Memory_rw, %Memory_write = firrtl.mem Undefined
      {
        depth = 12 : i64,
        name = "Memory",
        portNames = ["read", "rw", "write"],
        readLatency = 0 : i32,
        writeLatency = 1 : i32
      } :
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>,
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>,
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>

    %read_addr = firrtl.subfield %Memory_read[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    firrtl.connect %read_addr, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %read_en = firrtl.subfield %Memory_read[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    firrtl.connect %read_en, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %read_clk = firrtl.subfield %Memory_read[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    firrtl.connect %read_clk, %clock : !firrtl.clock, !firrtl.clock
    %read_data = firrtl.subfield %Memory_read[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    %read_data_slice = firrtl.bits %read_data 7 to 3 : (!firrtl.uint<42>) -> !firrtl.uint<5>
    firrtl.connect %result_read, %read_data_slice : !firrtl.uint<5>, !firrtl.uint<5>

    // CHECK-DAG: [[RW_FIELD:%.+]] = firrtl.subfield %Memory_rw[wdata] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<10>, wmode: uint<1>, wdata: uint<10>, wmask: uint<1>>
    // CHECK-DAG: [[RW_SLICE_LO:%.+]] = firrtl.bits %in_data 7 to 3 : (!firrtl.uint<42>) -> !firrtl.uint<5>
    // CHECK-DAG: [[RW_SLICE_HI:%.+]] = firrtl.bits %in_data 24 to 20 : (!firrtl.uint<42>) -> !firrtl.uint<5>
    // CHECK-DAG: [[RW_SLICE_JOIN:%.+]] = firrtl.cat [[RW_SLICE_HI]], [[RW_SLICE_LO]] : (!firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<10>
    // CHECK-DAG: firrtl.strictconnect [[RW_FIELD]], [[RW_SLICE_JOIN]] : !firrtl.uint<10>

    // CHECK-DAG: [[W_FIELD:%.+]] = firrtl.subfield %Memory_write[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<10>, mask: uint<1>>
    // CHECK-DAG: [[W_SLICE_LO:%.+]] = firrtl.bits %in_data 7 to 3 : (!firrtl.uint<42>) -> !firrtl.uint<5>
    // CHECK-DAG: [[W_SLICE_HI:%.+]] = firrtl.bits %in_data 24 to 20 : (!firrtl.uint<42>) -> !firrtl.uint<5>
    // CHECK-DAG: [[W_SLICE_JOIN:%.+]] = firrtl.cat [[W_SLICE_HI]], [[W_SLICE_LO]] : (!firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<10>
    // CHECK-DAG: firrtl.strictconnect [[W_FIELD]], [[W_SLICE_JOIN]] : !firrtl.uint<10>

    %rw_addr = firrtl.subfield %Memory_rw[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    firrtl.connect %rw_addr, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %rw_en = firrtl.subfield %Memory_rw[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    firrtl.connect %rw_en, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %rw_clk = firrtl.subfield %Memory_rw[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    firrtl.connect %rw_clk, %clock : !firrtl.clock, !firrtl.clock
    %rw_rdata = firrtl.subfield %Memory_rw[rdata] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    %rw_rdata_slice = firrtl.bits %rw_rdata 24 to 20 : (!firrtl.uint<42>) -> !firrtl.uint<5>
    firrtl.connect %result_rw, %rw_rdata_slice : !firrtl.uint<5>, !firrtl.uint<5>
    %rw_wmode = firrtl.subfield %Memory_rw[wmode] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    firrtl.connect %rw_wmode, %wmode_rw : !firrtl.uint<1>, !firrtl.uint<1>
    %rw_wdata = firrtl.subfield %Memory_rw[wdata] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    firrtl.connect %rw_wdata, %in_data : !firrtl.uint<42>, !firrtl.uint<42>
    %rw_wmask = firrtl.subfield %Memory_rw[wmask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, rdata flip: uint<42>, wmode: uint<1>, wdata: uint<42>, wmask: uint<1>>
    firrtl.connect %rw_wmask, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>


    %write_addr = firrtl.subfield %Memory_write[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    firrtl.connect %write_addr, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %write_en = firrtl.subfield %Memory_write[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    firrtl.connect %write_en, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %write_clk = firrtl.subfield %Memory_write[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    firrtl.connect %write_clk, %clock : !firrtl.clock, !firrtl.clock
    %write_data = firrtl.subfield %Memory_write[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    firrtl.connect %write_data, %in_data : !firrtl.uint<42>, !firrtl.uint<42>
    %write_mask = firrtl.subfield %Memory_write[mask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    firrtl.connect %write_mask, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "UnusedBitsAtEnd" {
  firrtl.module public @UnusedBitsAtEnd(
      in %clock: !firrtl.clock,
      in %addr: !firrtl.uint<4>,
      in %in_data: !firrtl.uint<42>,
      out %result_read: !firrtl.uint<5>) {

    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>

    // CHECK: %Memory_read, %Memory_write = firrtl.mem Undefined
    // CHECK-SAME: !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<5>>
    // CHECK-SAME: !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<5>, mask: uint<1>>
    %Memory_read, %Memory_write = firrtl.mem Undefined
      {
        depth = 12 : i64,
        name = "Memory",
        portNames = ["read", "write"],
        readLatency = 0 : i32,
        writeLatency = 1 : i32
      } :
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>,
        !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>

    // CHECK: [[RDATA:%.+]] = firrtl.subfield %Memory_read[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<5>>
    // CHECK: firrtl.strictconnect %result_read, [[RDATA]] : !firrtl.uint<5>
    %read_addr = firrtl.subfield %Memory_read[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    firrtl.connect %read_addr, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %read_en = firrtl.subfield %Memory_read[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    firrtl.connect %read_en, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %read_clk = firrtl.subfield %Memory_read[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    firrtl.connect %read_clk, %clock : !firrtl.clock, !firrtl.clock
    %read_data = firrtl.subfield %Memory_read[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    %read_data_slice = firrtl.bits %read_data 41 to 37 : (!firrtl.uint<42>) -> !firrtl.uint<5>
    firrtl.connect %result_read, %read_data_slice : !firrtl.uint<5>, !firrtl.uint<5>

    // CHECK: firrtl.bits %in_data 41 to 37 : (!firrtl.uint<42>) -> !firrtl.uint<5>
    %write_addr = firrtl.subfield %Memory_write[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    firrtl.connect %write_addr, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %write_en = firrtl.subfield %Memory_write[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    firrtl.connect %write_en, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %write_clk = firrtl.subfield %Memory_write[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    firrtl.connect %write_clk, %clock : !firrtl.clock, !firrtl.clock
    %write_data = firrtl.subfield %Memory_write[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    firrtl.connect %write_data, %in_data : !firrtl.uint<42>, !firrtl.uint<42>
    %write_mask = firrtl.subfield %Memory_write[mask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    firrtl.connect %write_mask, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "OneAddressMasked" {
  firrtl.module public @OneAddressMasked(
      in %clock: !firrtl.clock,
      in %addr: !firrtl.uint<1>,
      in %in_data: !firrtl.uint<32>,
      in %in_mask: !firrtl.uint<2>,
      in %in_wen: !firrtl.uint<1>,
      out %result_read: !firrtl.uint<32>) {


    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>

    %Memory_read, %Memory_write = firrtl.mem Undefined
      {
        depth = 1 : i64,
        name = "Memory",
        portNames = ["read", "write"],
        readLatency = 0 : i32,
        writeLatency = 1 : i32
      } :
        !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>,
        !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<2>>
    // CHECK: %Memory = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<32>

    // CHECK: firrtl.strictconnect %result_read, %Memory : !firrtl.uint<32>

    %read_addr = firrtl.subfield %Memory_read[addr] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>
    firrtl.connect %read_addr, %addr : !firrtl.uint<1>, !firrtl.uint<1>
    %read_en = firrtl.subfield %Memory_read[en] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>
    firrtl.connect %read_en, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %read_clk = firrtl.subfield %Memory_read[clk] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>
    firrtl.connect %read_clk, %clock : !firrtl.clock, !firrtl.clock
    %read_data = firrtl.subfield %Memory_read[data] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>
    firrtl.connect %result_read, %read_data : !firrtl.uint<32>, !firrtl.uint<32>

    // CHECK: [[DATA_0:%.+]] = firrtl.bits %in_data 15 to 0 : (!firrtl.uint<32>) -> !firrtl.uint<16>
    // CHECK: [[NEXT_0:%.+]] = firrtl.bits %Memory 15 to 0 : (!firrtl.uint<32>) -> !firrtl.uint<16>
    // CHECK: [[MASK_0:%.+]] = firrtl.bits %in_mask 0 to 0 : (!firrtl.uint<2>) -> !firrtl.uint<1>
    // CHECK: [[CHUNK_0:%.+]] = firrtl.mux([[MASK_0]], [[DATA_0]], [[NEXT_0]]) : (!firrtl.uint<1>, !firrtl.uint<16>, !firrtl.uint<16>) -> !firrtl.uint<16>
    // CHECK: [[DATA_1:%.+]] = firrtl.bits %in_data 31 to 16 : (!firrtl.uint<32>) -> !firrtl.uint<16>
    // CHECK: [[NEXT_1:%.+]] = firrtl.bits %Memory 31 to 16 : (!firrtl.uint<32>) -> !firrtl.uint<16>
    // CHECK: [[MASK_1:%.+]] = firrtl.bits %in_mask 1 to 1 : (!firrtl.uint<2>) -> !firrtl.uint<1>
    // CHECK: [[CHUNK_1:%.+]] = firrtl.mux([[MASK_1]], [[DATA_1]], [[NEXT_1]]) : (!firrtl.uint<1>, !firrtl.uint<16>, !firrtl.uint<16>) -> !firrtl.uint<16>
    // CHECK: [[NEXT:%.+]] = firrtl.cat [[CHUNK_1]], [[CHUNK_0]] : (!firrtl.uint<16>, !firrtl.uint<16>) -> !firrtl.uint<32>
    // CHECK: [[NEXT_EN:%.+]] = firrtl.mux(%in_wen, [[NEXT]], %Memory) : (!firrtl.uint<1>, !firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<32>
    // CHECK: firrtl.strictconnect %Memory, [[NEXT_EN]] : !firrtl.uint<32>

    %write_addr = firrtl.subfield %Memory_write[addr] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<2>>
    firrtl.connect %write_addr, %addr : !firrtl.uint<1>, !firrtl.uint<1>
    %write_en = firrtl.subfield %Memory_write[en] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<2>>
    firrtl.connect %write_en, %in_wen : !firrtl.uint<1>, !firrtl.uint<1>
    %write_clk = firrtl.subfield %Memory_write[clk] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<2>>
    firrtl.connect %write_clk, %clock : !firrtl.clock, !firrtl.clock
    %write_data = firrtl.subfield %Memory_write[data] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<2>>
    firrtl.connect %write_data, %in_data : !firrtl.uint<32>, !firrtl.uint<32>
    %write_mask = firrtl.subfield %Memory_write[mask] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<2>>
    firrtl.connect %write_mask, %in_mask : !firrtl.uint<2>, !firrtl.uint<2>
  }
}

// -----

firrtl.circuit "OneAddressNoMask" {
  firrtl.module public @OneAddressNoMask(
      in %clock: !firrtl.clock,
      in %addr: !firrtl.uint<1>,
      in %in_data: !firrtl.uint<32>,
      in %wmode_rw: !firrtl.uint<1>,
      in %in_wen: !firrtl.uint<1>,
      in %in_rwen: !firrtl.uint<1>,
      out %result_read: !firrtl.uint<32>,
      out %result_rw: !firrtl.uint<32>) {

    // Pipeline the inputs.
    // TODO: It would be good to de-duplicate these either in the pass or in a canonicalizer.

    // CHECK: %Memory_write_en_0 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %Memory_write_en_0, %in_wen : !firrtl.uint<1>
    // CHECK: %Memory_write_en_1 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %Memory_write_en_1, %Memory_write_en_0 : !firrtl.uint<1>
    // CHECK: %Memory_write_en_2 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %Memory_write_en_2, %Memory_write_en_1 : !firrtl.uint<1>

    // CHECK: %Memory_write_data_0 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<32>
    // CHECK: firrtl.strictconnect %Memory_write_data_0, %in_data : !firrtl.uint<32>
    // CHECK: %Memory_write_data_1 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<32>
    // CHECK: firrtl.strictconnect %Memory_write_data_1, %Memory_write_data_0 : !firrtl.uint<32>
    // CHECK: %Memory_write_data_2 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<32>
    // CHECK: firrtl.strictconnect %Memory_write_data_2, %Memory_write_data_1 : !firrtl.uint<32>

    // CHECK: %Memory_rw_wdata_0 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<32>
    // CHECK: firrtl.strictconnect %Memory_rw_wdata_0, %in_data : !firrtl.uint<32>
    // CHECK: %Memory_rw_wdata_1 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<32>
    // CHECK: firrtl.strictconnect %Memory_rw_wdata_1, %Memory_rw_wdata_0 : !firrtl.uint<32>
    // CHECK: %Memory_rw_wdata_2 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<32>
    // CHECK: firrtl.strictconnect %Memory_rw_wdata_2, %Memory_rw_wdata_1 : !firrtl.uint<32>

    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>

    %Memory_read, %Memory_rw, %Memory_write = firrtl.mem Undefined
      {
        depth = 1 : i64,
        name = "Memory",
        portNames = ["read", "rw", "write"],
        readLatency = 2 : i32,
        writeLatency = 4 : i32
      } :
        !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>,
        !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, rdata flip: uint<32>, wmode: uint<1>, wdata: uint<32>, wmask: uint<1>>,
        !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>

    // CHECK: %Memory = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<32>

    // CHECK: firrtl.strictconnect %result_read, %Memory : !firrtl.uint<32>
    %read_addr = firrtl.subfield %Memory_read[addr] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>
    firrtl.connect %read_addr, %addr : !firrtl.uint<1>, !firrtl.uint<1>
    %read_en = firrtl.subfield %Memory_read[en] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>
    firrtl.connect %read_en, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %read_clk = firrtl.subfield %Memory_read[clk] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>
    firrtl.connect %read_clk, %clock : !firrtl.clock, !firrtl.clock
    %read_data = firrtl.subfield %Memory_read[data] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<32>>
    firrtl.connect %result_read, %read_data : !firrtl.uint<32>, !firrtl.uint<32>

    // CHECK: firrtl.strictconnect %result_rw, %Memory : !firrtl.uint<32>
    %rw_addr = firrtl.subfield %Memory_rw[addr] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, rdata flip: uint<32>, wmode: uint<1>, wdata: uint<32>, wmask: uint<1>>
    firrtl.connect %rw_addr, %addr : !firrtl.uint<1>, !firrtl.uint<1>
    %rw_en = firrtl.subfield %Memory_rw[en] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, rdata flip: uint<32>, wmode: uint<1>, wdata: uint<32>, wmask: uint<1>>
    firrtl.connect %rw_en, %in_rwen : !firrtl.uint<1>, !firrtl.uint<1>
    %rw_clk = firrtl.subfield %Memory_rw[clk] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, rdata flip: uint<32>, wmode: uint<1>, wdata: uint<32>, wmask: uint<1>>
    firrtl.connect %rw_clk, %clock : !firrtl.clock, !firrtl.clock
    %rw_rdata = firrtl.subfield %Memory_rw[rdata] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, rdata flip: uint<32>, wmode: uint<1>, wdata: uint<32>, wmask: uint<1>>
    firrtl.connect %result_rw, %rw_rdata : !firrtl.uint<32>, !firrtl.uint<32>
    %rw_wmode = firrtl.subfield %Memory_rw[wmode] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, rdata flip: uint<32>, wmode: uint<1>, wdata: uint<32>, wmask: uint<1>>
    firrtl.connect %rw_wmode, %wmode_rw : !firrtl.uint<1>, !firrtl.uint<1>
    %rw_wdata = firrtl.subfield %Memory_rw[wdata] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, rdata flip: uint<32>, wmode: uint<1>, wdata: uint<32>, wmask: uint<1>>
    firrtl.connect %rw_wdata, %in_data : !firrtl.uint<32>, !firrtl.uint<32>
    %rw_wmask = firrtl.subfield %Memory_rw[wmask] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, rdata flip: uint<32>, wmode: uint<1>, wdata: uint<32>, wmask: uint<1>>
    firrtl.connect %rw_wmask, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>

    // CHECK: [[WRITING:%.+]] = firrtl.and %in_rwen, %wmode_rw : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK: %Memory_rw_wen_0 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %Memory_rw_wen_0, [[WRITING]] : !firrtl.uint<1>
    // CHECK: %Memory_rw_wen_1 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %Memory_rw_wen_1, %Memory_rw_wen_0 : !firrtl.uint<1>
    // CHECK: %Memory_rw_wen_2 = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<1>
    // CHECK: firrtl.strictconnect %Memory_rw_wen_2, %Memory_rw_wen_1 : !firrtl.uint<1>
    // CHECK: [[WRITE_RW:%.+]] = firrtl.mux(%Memory_rw_wen_2, %Memory_rw_wdata_2, %Memory) : (!firrtl.uint<1>, !firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<32>
    // CHECK: [[WRITE_W:%.+]] = firrtl.mux(%Memory_write_en_2, %Memory_write_data_2, [[WRITE_RW]]) : (!firrtl.uint<1>, !firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<32>
    // CHECK: firrtl.strictconnect %Memory, [[WRITE_W]] : !firrtl.uint<32>
    %write_addr = firrtl.subfield %Memory_write[addr] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    firrtl.connect %write_addr, %addr : !firrtl.uint<1>, !firrtl.uint<1>
    %write_en = firrtl.subfield %Memory_write[en] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    firrtl.connect %write_en, %in_wen : !firrtl.uint<1>, !firrtl.uint<1>
    %write_clk = firrtl.subfield %Memory_write[clk] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    firrtl.connect %write_clk, %clock : !firrtl.clock, !firrtl.clock
    %write_data = firrtl.subfield %Memory_write[data] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    firrtl.connect %write_data, %in_data : !firrtl.uint<32>, !firrtl.uint<32>
    %write_mask = firrtl.subfield %Memory_write[mask] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>
    firrtl.connect %write_mask, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

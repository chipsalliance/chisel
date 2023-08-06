// RUN: circt-opt -canonicalize %s | FileCheck %s

hw.module.extern @Observe(%x: i32)

// CHECK-LABEL: @FirReg
hw.module @FirReg(%clk: i1, %in: i32) {
  %false = hw.constant false
  %true = hw.constant true

  // Registers that update to themselves should be replaced with a constant 0.
  %reg0 = seq.firreg %reg0 clock %clk : i32
  hw.instance "reg0" @Observe(x: %reg0: i32) -> ()
  // CHECK: hw.instance "reg0" @Observe(x: %c0_i32: i32) -> ()

  // Registers that are never clocked should be replaced with a constant 0.
  %reg1a = seq.firreg %in clock %false : i32
  %reg1b = seq.firreg %in clock %true : i32
  hw.instance "reg1a" @Observe(x: %reg1a: i32) -> ()
  hw.instance "reg1b" @Observe(x: %reg1b: i32) -> ()
  // CHECK: hw.instance "reg1a" @Observe(x: %c0_i32: i32) -> ()
  // CHECK: hw.instance "reg1b" @Observe(x: %c0_i32: i32) -> ()
}

// Should not optimize away the register if it has a symbol.
// CHECK-LABEL: @FirRegSymbol
hw.module @FirRegSymbol(%clk: i1) -> (out : i32) {
  // CHECK: %reg = seq.firreg %reg clock %clk sym @reg : i32
  // CHECK: hw.output %reg : i32
  %reg = seq.firreg %reg clock %clk sym @reg : i32
  hw.output %reg : i32
}

// CHECK-LABEL: @FirRegReset
hw.module @FirRegReset(%clk: i1, %in: i32, %r : i1, %v : i32) {
  %false = hw.constant false
  %true = hw.constant true

  // Registers that update to themselves should be replaced with their reset
  // value.
  %reg0 = seq.firreg %reg0 clock %clk reset sync %r, %v : i32
  hw.instance "reg0" @Observe(x: %reg0: i32) -> ()
  // CHECK: hw.instance "reg0" @Observe(x: %v: i32) -> ()

  // Registers that never reset should drop their reset value.
  %reg1 = seq.firreg %in clock %clk reset sync %false, %v : i32
  hw.instance "reg1" @Observe(x: %reg1: i32) -> ()
  // CHECK: %reg1 = seq.firreg %in clock %clk : i32
  // CHECK: hw.instance "reg1" @Observe(x: %reg1: i32) -> ()

  // Registers that are permanently reset should be replaced with their reset
  // value.
  %reg2a = seq.firreg %in clock %clk reset sync %true, %v : i32
  %reg2b = seq.firreg %in clock %clk reset async %true, %v : i32
  hw.instance "reg2a" @Observe(x: %reg2a: i32) -> ()
  hw.instance "reg2b" @Observe(x: %reg2b: i32) -> ()
  // CHECK: hw.instance "reg2a" @Observe(x: %v: i32) -> ()
  // CHECK: hw.instance "reg2b" @Observe(x: %v: i32) -> ()

  // Registers that are never clocked should be replaced with their reset value.
  %reg3a = seq.firreg %in clock %false reset sync %r, %v : i32
  %reg3b = seq.firreg %in clock %true reset sync %r, %v : i32
  hw.instance "reg3a" @Observe(x: %reg3a: i32) -> ()
  hw.instance "reg3b" @Observe(x: %reg3b: i32) -> ()
  // CHECK: hw.instance "reg3a" @Observe(x: %v: i32) -> ()
  // CHECK: hw.instance "reg3b" @Observe(x: %v: i32) -> ()
}

// CHECK-LABEL: @FirRegAggregate
hw.module @FirRegAggregate(%clk: i1) -> (out : !hw.struct<foo: i32>) {
  // TODO: Use constant aggregate attribute once supported.
  // CHECK:      %c0_i32 = hw.constant 0 : i32
  // CHECK-NEXT: %0 = hw.bitcast %c0_i32 : (i32) -> !hw.struct<foo: i32>
  // CHECK-NEXT: hw.output %0
  %reg = seq.firreg %reg clock %clk : !hw.struct<foo: i32>
  hw.output %reg : !hw.struct<foo: i32>
}

// CHECK-LABEL: @UninitializedArrayElement
hw.module @UninitializedArrayElement(%a: i1, %clock: i1) -> (b: !hw.array<2xi1>) {
  // CHECK:      %false = hw.constant false
  // CHECK-NEXT: %0 = hw.array_create %false, %a : i1
  // CHECK-NEXT: %r = seq.firreg %0 clock %clock : !hw.array<2xi1>
  // CHECK-NEXT: hw.output %r : !hw.array<2xi1>
  %true = hw.constant true
  %r = seq.firreg %1 clock %clock : !hw.array<2xi1>
  %0 = hw.array_get %r[%true] : !hw.array<2xi1>, i1
  %1 = hw.array_create %0, %a : i1
  hw.output %r : !hw.array<2xi1>
}

// CHECK-LABEL: hw.module @ClockGate
hw.module @ClockGate(%clock: i1, %enable: i1, %enable2 : i1, %testEnable: i1) {
  // CHECK-NEXT: hw.constant false
  %false = hw.constant false
  %true = hw.constant true

  // CHECK-NEXT: %zeroClock = hw.wire %false sym @zeroClock
  %0 = seq.clock_gate %false, %enable
  %zeroClock = hw.wire %0 sym @zeroClock : i1

  // CHECK-NEXT: %alwaysOff1 = hw.wire %false sym @alwaysOff1
  // CHECK-NEXT: %alwaysOff2 = hw.wire %false sym @alwaysOff2
  %1 = seq.clock_gate %clock, %false
  %2 = seq.clock_gate %clock, %false, %false
  %alwaysOff1 = hw.wire %1 sym @alwaysOff1 : i1
  %alwaysOff2 = hw.wire %2 sym @alwaysOff2 : i1

  // CHECK-NEXT: %alwaysOn1 = hw.wire %clock sym @alwaysOn1
  // CHECK-NEXT: %alwaysOn2 = hw.wire %clock sym @alwaysOn2
  // CHECK-NEXT: %alwaysOn3 = hw.wire %clock sym @alwaysOn3
  %3 = seq.clock_gate %clock, %true
  %4 = seq.clock_gate %clock, %true, %testEnable
  %5 = seq.clock_gate %clock, %enable, %true
  %alwaysOn1 = hw.wire %3 sym @alwaysOn1 : i1
  %alwaysOn2 = hw.wire %4 sym @alwaysOn2 : i1
  %alwaysOn3 = hw.wire %5 sym @alwaysOn3 : i1

  // CHECK-NEXT: [[TMP:%.+]] = seq.clock_gate %clock, %enable
  // CHECK-NEXT: %dropTestEnable = hw.wire [[TMP]] sym @dropTestEnable
  %6 = seq.clock_gate %clock, %enable, %false
  %dropTestEnable = hw.wire %6 sym @dropTestEnable : i1

  // CHECK-NEXT: [[TCG1:%.+]] = seq.clock_gate %clock, %enable
  // CHECK-NEXT: %transitiveClock1 = hw.wire [[TCG1]] sym @transitiveClock1  : i1
  %7 = seq.clock_gate %clock, %enable
  %8 = seq.clock_gate %clock, %enable
  %transitiveClock1 = hw.wire %7 sym @transitiveClock1 : i1

  // CHECK-NEXT: [[TCG2:%.+]] = seq.clock_gate %clock, %enable, %testEnable
  // CHECK-NEXT: [[TCG3:%.+]] = seq.clock_gate [[TCG2]], %enable
  // CHECK-NEXT: %transitiveClock2 = hw.wire [[TCG3]] sym @transitiveClock2  : i1
  %9 = seq.clock_gate %clock, %enable, %testEnable
  %10 = seq.clock_gate %9, %enable2 
  %11 = seq.clock_gate %10, %enable, %testEnable
  %transitiveClock2 = hw.wire %11 sym @transitiveClock2 : i1
}

// CHECK-LABEL: @FirMem
hw.module @FirMem(%addr: i4, %clock: i1, %data: i42) -> (out: i42) {
  %true = hw.constant true
  %false = hw.constant false
  %c0_i3 = hw.constant 0 : i3
  %c-1_i3 = hw.constant -1 : i3

  // CHECK: [[MEM:%.+]] = seq.firmem
  %0 = seq.firmem 0, 1, undefined, undefined : <12 x 42, mask 3>

  // CHECK-NEXT: seq.firmem.read_port [[MEM]][%addr], clock %clock :
  %1 = seq.firmem.read_port %0[%addr], clock %clock enable %true : <12 x 42, mask 3>

  // CHECK-NEXT: seq.firmem.write_port %0[%addr] = %data, clock %clock {w0}
  seq.firmem.write_port %0[%addr] = %data, clock %clock enable %true {w0} : <12 x 42, mask 3>
  // CHECK-NOT: {w1}
  seq.firmem.write_port %0[%addr] = %data, clock %clock enable %false {w1} : <12 x 42, mask 3>
  // CHECK-NEXT: seq.firmem.write_port %0[%addr] = %data, clock %clock {w2}
  seq.firmem.write_port %0[%addr] = %data, clock %clock mask %c-1_i3 {w2} : <12 x 42, mask 3>, i3
  // CHECK-NOT: {w3}
  seq.firmem.write_port %0[%addr] = %data, clock %clock mask %c0_i3 {w3} : <12 x 42, mask 3>, i3
  // CHECK-NOT: {w4}
  seq.firmem.write_port %0[%addr] = %data, clock %true {w4} : <12 x 42, mask 3>
  // CHECK-NOT: {w5}
  seq.firmem.write_port %0[%addr] = %data, clock %false {w5} : <12 x 42, mask 3>

  // CHECK-NEXT: seq.firmem.read_write_port [[MEM]][%addr] = %data if %true, clock %clock {rw0}
  %2 = seq.firmem.read_write_port %0[%addr] = %data if %true, clock %clock enable %true {rw0} : <12 x 42, mask 3>
  // CHECK-NEXT: seq.firmem.read_port [[MEM]][%addr], clock %clock enable %false {rw1}
  %3 = seq.firmem.read_write_port %0[%addr] = %data if %true, clock %clock enable %false {rw1} : <12 x 42, mask 3>
  // CHECK-NEXT: seq.firmem.read_write_port [[MEM]][%addr] = %data if %true, clock %clock {rw2}
  %4 = seq.firmem.read_write_port %0[%addr] = %data if %true, clock %clock mask %c-1_i3 {rw2} : <12 x 42, mask 3>, i3
  // CHECK-NEXT: seq.firmem.read_port [[MEM]][%addr], clock %clock {rw3}
  %5 = seq.firmem.read_write_port %0[%addr] = %data if %true, clock %clock mask %c0_i3 {rw3} : <12 x 42, mask 3>, i3
  // CHECK-NEXT: seq.firmem.read_port [[MEM]][%addr], clock %clock {rw4}
  %6 = seq.firmem.read_write_port %0[%addr] = %data if %false, clock %clock {rw4} : <12 x 42, mask 3>
  // CHECK-NEXT: seq.firmem.read_port [[MEM]][%addr], clock %true {rw5}
  %7 = seq.firmem.read_write_port %0[%addr] = %data if %true, clock %true {rw5} : <12 x 42, mask 3>
  // CHECK-NEXT: seq.firmem.read_port [[MEM]][%addr], clock %false {rw6}
  %8 = seq.firmem.read_write_port %0[%addr] = %data if %true, clock %false {rw6} : <12 x 42, mask 3>

  %9 = comb.xor %1, %2, %3, %4, %5, %6, %7, %8 : i42
  hw.output %9 : i42
}

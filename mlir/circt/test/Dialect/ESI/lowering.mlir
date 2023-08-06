// RUN: circt-opt %s --lower-esi-to-physical -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s --lower-esi-ports -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck --check-prefix=IFACE %s
// RUN: circt-opt %s --lower-esi-to-physical --lower-esi-ports --hw-flatten-io --lower-esi-to-hw -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck --check-prefix=HW %s

hw.module.extern @Sender(%clk: i1) -> (x: !esi.channel<i4>, y: i8) attributes {esi.bundle}
hw.module.extern @ArrSender() -> (x: !esi.channel<!hw.array<4xi64>>) attributes {esi.bundle}
hw.module.extern @Reciever(%a: !esi.channel<i4>, %clk: i1) attributes {esi.bundle}
hw.module.extern @i0SenderReceiver(%in: !esi.channel<i0>) -> (out: !esi.channel<i0>)

// CHECK-LABEL: hw.module.extern @Sender(%clk: i1) -> (x: !esi.channel<i4>, y: i8)
// CHECK-LABEL: hw.module.extern @Reciever(%a: !esi.channel<i4>, %clk: i1)

// IFACE-LABEL: sv.interface @IValidReady_i4 {
// IFACE-NEXT:    sv.interface.signal @valid : i1
// IFACE-NEXT:    sv.interface.signal @ready : i1
// IFACE-NEXT:    sv.interface.signal @data : i4
// IFACE-NEXT:    sv.interface.modport @sink (input @ready, output @valid, output @data)
// IFACE-NEXT:    sv.interface.modport @source (input @valid, input @data, output @ready)
// IFACE-LABEL: sv.interface @IValidReady_ArrayOf4xi64 {
// IFACE-NEXT:    sv.interface.signal @valid : i1
// IFACE-NEXT:    sv.interface.signal @ready : i1
// IFACE-NEXT:    sv.interface.signal @data : !hw.array<4xi64>
// IFACE-NEXT:    sv.interface.modport @sink  (input @ready, output @valid, output @data)
// IFACE-NEXT:    sv.interface.modport @source  (input @valid, input @data, output @ready)
// IFACE-LABEL: hw.module.extern @Sender(%clk: i1, %x: !sv.modport<@IValidReady_i4::@sink>) -> (y: i8)
// IFACE-LABEL: hw.module.extern @ArrSender(%x: !sv.modport<@IValidReady_ArrayOf4xi64::@sink>)
// IFACE-LABEL: hw.module.extern @Reciever(%a: !sv.modport<@IValidReady_i4::@source>, %clk: i1)
// IFACE-LABEL: hw.module.extern @i0SenderReceiver(%in: i0, %in_valid: i1, %out_ready: i1) -> (in_ready: i1, out: i0, out_valid: i1)

hw.module @test(%clk:i1, %rst:i1) {

  %esiChan2, %0 = hw.instance "sender2" @Sender(clk: %clk: i1) -> (x: !esi.channel<i4>, y: i8)
  %bufferedChan2 = esi.buffer %clk, %rst, %esiChan2 { stages = 4 } : i4
  hw.instance "recv2" @Reciever (a: %bufferedChan2: !esi.channel<i4>, clk: %clk: i1) -> ()

  // CHECK:      %sender2.x, %sender2.y = hw.instance "sender2" @Sender(clk: %clk: i1) -> (x: !esi.channel<i4>, y: i8)
  // CHECK-NEXT:  %0 = esi.stage %clk, %rst, %sender2.x : i4
  // CHECK-NEXT:  %1 = esi.stage %clk, %rst, %0 : i4
  // CHECK-NEXT:  %2 = esi.stage %clk, %rst, %1 : i4
  // CHECK-NEXT:  %3 = esi.stage %clk, %rst, %2 : i4
  // CHECK-NEXT:  hw.instance "recv2" @Reciever(a: %3: !esi.channel<i4>, clk: %clk: i1) -> ()

  // IFACE-LABEL: hw.module @test(%clk: i1, %rst: i1) {
  // IFACE-NEXT:    %i4FromSender2 = sv.interface.instance : !sv.interface<@IValidReady_i4>
  // IFACE-NEXT:    %[[#modport1:]] = sv.modport.get %i4FromSender2 @source : !sv.interface<@IValidReady_i4> -> !sv.modport<@IValidReady_i4::@source>
  // IFACE-NEXT:    %[[#channel:]] = esi.wrap.iface %[[#modport1:]] : !sv.modport<@IValidReady_i4::@source> -> !esi.channel<i4>
  // IFACE-NEXT:    %[[#modport2:]] = sv.modport.get %i4FromSender2 @sink : !sv.interface<@IValidReady_i4> -> !sv.modport<@IValidReady_i4::@sink>
  // IFACE-NEXT:    %sender2.y = hw.instance "sender2" @Sender(clk: %clk: i1, x: %2: !sv.modport<@IValidReady_i4::@sink>) -> (y: i8)
  // IFACE-NEXT:    %[[#buffer:]] = esi.buffer %clk, %rst, %[[#channel:]] {stages = 4 : i64} : i4
  // IFACE-NEXT:    %i4ToRecv2 = sv.interface.instance : !sv.interface<@IValidReady_i4>
  // IFACE-NEXT:    %[[#modport3:]] = sv.modport.get %i4ToRecv2 @sink : !sv.interface<@IValidReady_i4> -> !sv.modport<@IValidReady_i4::@sink>
  // IFACE-NEXT:    esi.unwrap.iface %[[#buffer:]] into %[[#modport3:]] : (!esi.channel<i4>, !sv.modport<@IValidReady_i4::@sink>)
  // IFACE-NEXT:    %[[#modport4:]] = sv.modport.get %i4ToRecv2 @source : !sv.interface<@IValidReady_i4> -> !sv.modport<@IValidReady_i4::@source>
  // IFACE-NEXT:    hw.instance "recv2" @Reciever(a: %[[#modport4:]]: !sv.modport<@IValidReady_i4::@source>, clk: %clk: i1) -> ()

  // After all 3 ESI lowering passes, there shouldn't be any ESI constructs!
  // HW-NOT: esi
}

hw.module @add11(%clk: i1, %ints: !esi.channel<i32>) -> (mutatedInts: !esi.channel<i32>, c4: i4) {
  %i, %i_valid = esi.unwrap.vr %ints, %rdy : i32
  %c11 = hw.constant 11 : i32
  %m = comb.add %c11, %i : i32
  %mutInts, %rdy = esi.wrap.vr %m, %i_valid : i32
  %c4 = hw.constant 0 : i4
  hw.output %mutInts, %c4 : !esi.channel<i32>, i4
}
// HW-LABEL: hw.module @add11(%clk: i1, %ints: i32, %ints_valid: i1, %mutatedInts_ready: i1) -> (ints_ready: i1, mutatedInts: i32, mutatedInts_valid: i1, c4: i4) {
// HW:   %{{.+}} = hw.constant 11 : i32
// HW:   [[RES0:%.+]] = comb.add %ints, %{{.+}} : i32
// HW:   %{{.+}} = hw.constant 0 : i4
// HW:   hw.output %mutatedInts_ready, [[RES0]], %ints_valid, %{{.+}} : i1, i32, i1, i4

hw.module @InternRcvr(%in: !esi.channel<!hw.array<4xi8>>) {}

hw.module @test2(%clk:i1, %rst:i1) {
  %ints, %c4 = hw.instance "adder" @add11(clk: %clk: i1, ints: %ints: !esi.channel<i32>) -> (mutatedInts: !esi.channel<i32>, c4: i4)

  %nullBit = esi.null : !esi.channel<i4>
  hw.instance "nullRcvr" @Reciever(a: %nullBit: !esi.channel<i4>, clk: %clk: i1) -> ()

  %nullArray = esi.null : !esi.channel<!hw.array<4xi8>>
  hw.instance "nullInternRcvr" @InternRcvr(in: %nullArray: !esi.channel<!hw.array<4xi8>>) -> ()
}
// HW-LABEL: hw.module @test2(%clk: i1, %rst: i1) {
// HW:   %adder.ints_ready, %adder.mutatedInts, %adder.mutatedInts_valid, %adder.c4 = hw.instance "adder" @add11(clk: %clk: i1, ints: %adder.mutatedInts: i32, ints_valid: %adder.mutatedInts_valid: i1, mutatedInts_ready: %adder.ints_ready: i1) -> (ints_ready: i1, mutatedInts: i32, mutatedInts_valid: i1, c4: i4)
// HW:   [[ZERO:%.+]] = hw.bitcast %c0_i4 : (i4) -> i4
// HW:   sv.interface.signal.assign %i4ToNullRcvr(@IValidReady_i4::@data) = [[ZERO]] : i4
// HW:   [[ZM:%.+]] = sv.modport.get %{{.+}} @source : !sv.interface<@IValidReady_i4> -> !sv.modport<@IValidReady_i4::@source>
// HW:   hw.instance "nullRcvr" @Reciever(a: [[ZM]]: !sv.modport<@IValidReady_i4::@source>, clk: %clk: i1) -> ()
// HW:   %c0_i32 = hw.constant 0 : i32
// HW:   [[ZA:%.+]] = hw.bitcast %c0_i32 : (i32) -> !hw.array<4xi8>
// HW:   %nullInternRcvr.in_ready = hw.instance "nullInternRcvr" @InternRcvr(in: [[ZA]]: !hw.array<4xi8>, in_valid: %false_0: i1) -> (in_ready: i1)

hw.module @twoChannelArgs(%clk: i1, %ints: !esi.channel<i32>, %foo: !esi.channel<i7>) -> () {
  %rdy = hw.constant 1 : i1
  %i, %i_valid = esi.unwrap.vr %ints, %rdy : i32
  %i2, %i2_valid = esi.unwrap.vr %foo, %rdy : i7
}
// HW-LABEL: hw.module @twoChannelArgs(%clk: i1, %ints: i32, %ints_valid: i1, %foo: i7, %foo_valid: i1) -> (ints_ready: i1, foo_ready: i1)
// HW:   %true = hw.constant true
// HW:   hw.output %true, %true : i1, i1

// IFACE: %i1ToHandshake_fork0FromArg0 = sv.interface.instance : !sv.interface<@IValidReady_i1>
hw.module.extern @handshake_fork_1ins_2outs_ctrl(%in0: !esi.channel<i1>, %clock: i1, %reset: i1) -> (out0: !esi.channel<i1>, out1: !esi.channel<i1>) attributes {esi.bundle}
hw.module @test_constant(%arg0: !esi.channel<i1>, %clock: i1, %reset: i1) -> (outCtrl: !esi.channel<i1>) {
  %handshake_fork0.out0, %handshake_fork0.out1 = hw.instance "handshake_fork0" @handshake_fork_1ins_2outs_ctrl(in0: %arg0: !esi.channel<i1>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i1>, out1: !esi.channel<i1>)
  hw.output %handshake_fork0.out1 : !esi.channel<i1>
}

// HW-LABEL: hw.module @i0Typed(%a: i0, %a_valid: i1, %clk: i1, %rst: i1, %x_ready: i1) -> (a_ready: i1, x: i0, x_valid: i1) {
// HW:         %pipelineStage.a_ready, %pipelineStage.x, %pipelineStage.x_valid = hw.instance "pipelineStage" @ESI_PipelineStage1<WIDTH: ui32 = 0>(clk: %clk: i1, rst: %rst: i1, a: %a: i0, a_valid: %a_valid: i1, x_ready: %x_ready: i1) -> (a_ready: i1, x: i0, x_valid: i1)
// HW:         hw.output %pipelineStage.a_ready, %pipelineStage.x, %pipelineStage.x_valid : i1, i0, i1
// HW:       }
hw.module @i0Typed(%a: !esi.channel<i0>, %clk: i1, %rst: i1) -> (x: !esi.channel<i0>) {
  %0 = esi.buffer %clk, %rst, %a  : i0
  %i0Value, %valid = esi.unwrap.vr %0, %ready : i0
  %chanOutput, %ready = esi.wrap.vr %i0Value, %valid : i0
  hw.output %chanOutput : !esi.channel<i0>
}

// IFACE: hw.module @HandshakeToESIWrapper(%clock: i1, %reset: i1, %in_ctrl: i0, %in_ctrl_valid: i1, %in0_ld_data0: i32, %in0_ld_data0_valid: i1, %in1_ld_data0: i32, %in1_ld_data0_valid: i1, %out_ctrl_ready: i1, %in0_ld_addr0_ready: i1, %in1_ld_addr0_ready: i1, %out0_ready: i1) -> (in_ctrl_ready: i1, in0_ld_data0_ready: i1, in1_ld_data0_ready: i1, out_ctrl: i0, out_ctrl_valid: i1, in0_ld_addr0: i64, in0_ld_addr0_valid: i1, in1_ld_addr0: i64, in1_ld_addr0_valid: i1, out0: i32, out0_valid: i1)
hw.module @HandshakeToESIWrapper(%clock: i1, %reset: i1, %in_ctrl: !esi.channel<i0>, %in0_ld_data0: !esi.channel<i32>, %in1_ld_data0: !esi.channel<i32>) -> (out_ctrl: !esi.channel<i0>, in0_ld_addr0: !esi.channel<i64>, in1_ld_addr0: !esi.channel<i64>, out0: !esi.channel<i32>) {
  %i0 = hw.constant 0 : i0
  %c1 = hw.constant 1 : i1
  %c32 = hw.constant 1 : i32
  %c64 = hw.constant 1 : i64
  %chanOutput, %ready = esi.wrap.vr %i0, %c1 : i0
  %chanOutput_2, %ready_3 = esi.wrap.vr %c64, %c1 : i64
  %chanOutput_6, %ready_7 = esi.wrap.vr %c64, %c1 : i64
  %chanOutput_8, %ready_9 = esi.wrap.vr %c32, %c1 : i32
  hw.output %chanOutput, %chanOutput_2, %chanOutput_6, %chanOutput_8 : !esi.channel<i0>, !esi.channel<i64>, !esi.channel<i64>, !esi.channel<i32>
}

// IFACE: hw.module @ServiceWrapper(%clock: i1, %reset: i1, %ctrl: i0, %ctrl_valid: i1, %port0: i32, %port0_valid: i1, %port1: i32, %port1_valid: i1, %ctrl_ready: i1, %port0_ready: i1, %port1_ready: i1) -> (ctrl_ready: i1, port0_ready: i1, port1_ready: i1, ctrl: i0, ctrl_valid: i1, port0: i64, port0_valid: i1, port1: i64, port1_valid: i1)
hw.module @ServiceWrapper(%clock: i1, %reset: i1, %ctrl: !esi.channel<i0>, %port0: !esi.channel<i32>, %port1: !esi.channel<i32>) -> (ctrl: !esi.channel<i0>, port0: !esi.channel<i64>, port1: !esi.channel<i64>) {
  %HandshakeToESIWrapper.out_ctrl, %HandshakeToESIWrapper.in0_ld_addr0, %HandshakeToESIWrapper.in1_ld_addr0, %HandshakeToESIWrapper.out0 = hw.instance "HandshakeToESIWrapper" @HandshakeToESIWrapper(clock: %clock: i1, reset: %reset: i1, in_ctrl: %ctrl: !esi.channel<i0>, in0_ld_data0: %port0: !esi.channel<i32>, in1_ld_data0: %port1: !esi.channel<i32>) -> (out_ctrl: !esi.channel<i0>, in0_ld_addr0: !esi.channel<i64>, in1_ld_addr0: !esi.channel<i64>, out0: !esi.channel<i32>)
  hw.output %HandshakeToESIWrapper.out_ctrl, %HandshakeToESIWrapper.in0_ld_addr0, %HandshakeToESIWrapper.in1_ld_addr0 : !esi.channel<i0>, !esi.channel<i64>, !esi.channel<i64>
}

// IFACE-LABEL:  hw.module @i1Fifo0Loopback(%in: i3, %in_empty: i1, %out_rden: i1) -> (in_rden: i1, out: i3, out_empty: i1)
// IFACE-NEXT:     %chanOutput, %rden = esi.wrap.fifo %in, %in_empty : !esi.channel<i3, FIFO0>
// IFACE-NEXT:     %data, %empty = esi.unwrap.fifo %chanOutput, %out_rden : !esi.channel<i3, FIFO0>
// IFACE-NEXT:     hw.output %rden, %data, %empty : i1, i3, i1
// HW-LABEL:     hw.module @i1Fifo0Loopback(%in: i3, %in_empty: i1, %out_rden: i1) -> (in_rden: i1, out: i3, out_empty: i1)
// HW-NEXT:        hw.output %out_rden, %in, %in_empty : i1, i3, i1
hw.module @i1Fifo0Loopback(%in: !esi.channel<i3, FIFO0>) -> (out: !esi.channel<i3, FIFO0>) {
  hw.output %in : !esi.channel<i3, FIFO0>
}

// IFACE-LABEL:  hw.module @fifo0LoopbackTop()
// IFACE-NEXT:     %data, %empty = esi.unwrap.fifo %chanOutput, %foo.in_rden : !esi.channel<i3, FIFO0>
// IFACE-NEXT:     %chanOutput, %rden = esi.wrap.fifo %foo.out, %foo.out_empty : !esi.channel<i3, FIFO0>
// IFACE-NEXT:     %foo.in_rden, %foo.out, %foo.out_empty = hw.instance "foo" @i1Fifo0Loopback(in: %data: i3, in_empty: %empty: i1, out_rden: %rden: i1) -> (in_rden: i1, out: i3, out_empty: i1)
// HW-LABEL:     hw.module @fifo0LoopbackTop()
// HW-NEXT:        %foo.in_rden, %foo.out, %foo.out_empty = hw.instance "foo" @i1Fifo0Loopback(in: %foo.out: i3, in_empty: %foo.out_empty: i1, out_rden: %foo.in_rden: i1) -> (in_rden: i1, out: i3, out_empty: i1)
hw.module @fifo0LoopbackTop() -> () {
  %chan = hw.instance "foo" @i1Fifo0Loopback(in: %chan: !esi.channel<i3, FIFO0>) -> (out: !esi.channel<i3, FIFO0>)
}

// IFACE-LABEL:  hw.module @structFifo0Loopback(%in_in: !hw.struct<a: i3, b: i7>, %in_flatBroke_in: i1, %out_readEnable_in: i1) -> (in_readEnable: i1, out: !hw.struct<a: i3, b: i7>, out_flatBroke: i1)
// IFACE-NEXT:     %chanOutput, %rden = esi.wrap.fifo %in_in, %in_flatBroke_in : !esi.channel<!hw.struct<a: i3, b: i7>, FIFO0>
// IFACE-NEXT:     %data, %empty = esi.unwrap.fifo %chanOutput, %out_readEnable_in : !esi.channel<!hw.struct<a: i3, b: i7>, FIFO0>
// IFACE-NEXT:     hw.output %rden, %data, %empty : i1, !hw.struct<a: i3, b: i7>, i1
!st1 = !hw.struct<a: i3, b: i7>
hw.module @structFifo0Loopback(%in: !esi.channel<!st1, FIFO0>) -> (out: !esi.channel<!st1, FIFO0>)
    attributes {esi.portFlattenStructs, esi.portRdenSuffix="_readEnable",
                esi.portEmptySuffix="_flatBroke", esi.portInSuffix="_in"} {
  hw.output %in : !esi.channel<!st1, FIFO0>
}

// IFACE-LABEL:  hw.module @structFifo0LoopbackTop()
// IFACE-NEXT:    %data, %empty = esi.unwrap.fifo %chanOutput, %foo.in_readEnable : !esi.channel<!hw.struct<a: i3, b: i7>, FIFO0>
// IFACE-NEXT:    %chanOutput, %rden = esi.wrap.fifo %foo.out, %foo.out_flatBroke : !esi.channel<!hw.struct<a: i3, b: i7>, FIFO0>
// IFACE-NEXT:    %foo.in_readEnable, %foo.out, %foo.out_flatBroke = hw.instance "foo" @structFifo0Loopback(in_in: %data: !hw.struct<a: i3, b: i7>, in_flatBroke_in: %empty: i1, out_readEnable_in: %rden: i1) -> (in_readEnable: i1, out: !hw.struct<a: i3, b: i7>, out_flatBroke: i1)
// IFACE-NEXT:    hw.output
hw.module @structFifo0LoopbackTop() -> () {
  %chan = hw.instance "foo" @structFifo0Loopback(in: %chan: !esi.channel<!st1, FIFO0>) -> (out: !esi.channel<!st1, FIFO0>)
}

// IFACE-LABEL:  hw.module @i3LoopbackOddNames(%in: i3, %in_good: i1, %out_letErRip: i1) -> (in_letErRip_out: i1, out_out: i3, out_good_out: i1) attributes {esi.portFlattenStructs, esi.portOutSuffix = "_out", esi.portReadySuffix = "_letErRip", esi.portValidSuffix = "_good"} {
// IFACE-NEXT:    %chanOutput, %ready = esi.wrap.vr %in, %in_good : i3
// IFACE-NEXT:    %rawOutput, %valid = esi.unwrap.vr %chanOutput, %out_letErRip : i3
// IFACE-NEXT:    hw.output %ready, %rawOutput, %valid : i1, i3, i1
hw.module @i3LoopbackOddNames(%in: !esi.channel<i3>) -> (out: !esi.channel<i3>)
    attributes {esi.portFlattenStructs, esi.portValidSuffix="_good",
                esi.portReadySuffix="_letErRip", esi.portOutSuffix="_out"} {
  hw.output %in : !esi.channel<i3>
}

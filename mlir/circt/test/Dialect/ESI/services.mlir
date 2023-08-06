// RUN: circt-opt %s | circt-opt | FileCheck %s
// RUN: circt-opt --esi-connect-services  %s | circt-opt | FileCheck %s --check-prefix=CONN

// CHECK-LABEL: esi.service.decl @HostComms {
// CHECK:         esi.service.to_server @Send : !esi.channel<!esi.any>
// CHECK:         esi.service.to_client @Recv : !esi.channel<i8>
esi.service.decl @HostComms {
  esi.service.to_server @Send : !esi.channel<!esi.any>
  esi.service.to_client @Recv : !esi.channel<i8>
  esi.service.inout @ReqResp : !esi.channel<i8> -> !esi.channel<i16>
}

// CHECK-LABEL: hw.module @Top(%clk: i1, %rst: i1) {
// CHECK:         esi.service.instance impl as "cosim"(%clk, %rst) : (i1, i1) -> ()
// CHECK:         hw.instance "m1" @Loopback(clk: %clk: i1) -> ()

// CONN-LABEL: hw.module @Top(%clk: i1, %rst: i1) {
// CONN-DAG:     [[R2:%.+]] = esi.cosim %clk, %rst, %m1.loopback_fromhw, "m1.loopback_fromhw" : !esi.channel<i8> -> !esi.channel<i1>
// CONN-DAG:     [[R0:%.+]] = esi.null : !esi.channel<i1>
// CONN-DAG:     [[R1:%.+]] = esi.cosim %clk, %rst, [[R0]], "m1.loopback_tohw" : !esi.channel<i1> -> !esi.channel<i8>
// CONN:         %m1.loopback_fromhw = hw.instance "m1" @Loopback(clk: %clk: i1, loopback_tohw: [[R1]]: !esi.channel<i8>) -> (loopback_fromhw: !esi.channel<i8>)
hw.module @Top (%clk: i1, %rst: i1) {
  esi.service.instance impl as  "cosim" (%clk, %rst) : (i1, i1) -> ()
  hw.instance "m1" @Loopback (clk: %clk: i1) -> ()
}

// CHECK-LABEL: hw.module @Loopback(%clk: i1) {
// CHECK:         %0 = esi.service.req.to_client <@HostComms::@Recv>(["loopback_tohw"]) : !esi.channel<i8>
// CHECK:         esi.service.req.to_server %0 -> <@HostComms::@Send>(["loopback_fromhw"]) : !esi.channel<i8>

// CONN-LABEL: hw.module @Loopback(%clk: i1, %loopback_tohw: !esi.channel<i8>) -> (loopback_fromhw: !esi.channel<i8>) {
// CONN:         hw.output %loopback_tohw : !esi.channel<i8>
hw.module @Loopback (%clk: i1) -> () {
  %dataIn = esi.service.req.to_client <@HostComms::@Recv> (["loopback_tohw"]) : !esi.channel<i8>
  esi.service.req.to_server %dataIn -> <@HostComms::@Send> (["loopback_fromhw"]) : !esi.channel<i8>
}

// CONN-LABEL: hw.module @Top2(%clk: i1) -> (chksum: i8) {
// CONN:         [[r0:%.+]]:3 = esi.service.impl_req svc @HostComms impl as "topComms2"(%clk) : (i1) -> (i8, !esi.channel<i8>, !esi.channel<i8>) {
// CONN-DAG:       esi.service.req.to_client <@HostComms::@Recv>(["r1", "m1", "loopback_tohw"]) : !esi.channel<i8>
// CONN-DAG:       esi.service.req.to_client <@HostComms::@Recv>(["r1", "c1", "consumingFromChan"]) : !esi.channel<i8>
// CONN-DAG:       esi.service.req.to_server %r1.m1.loopback_fromhw -> <@HostComms::@Send>(["r1", "m1", "loopback_fromhw"]) : !esi.channel<i8>
// CONN-DAG:       esi.service.req.to_server %r1.p1.producedMsgChan -> <@HostComms::@Send>(["r1", "p1", "producedMsgChan"]) : !esi.channel<i8>
// CONN-DAG:       esi.service.req.to_server %r1.p2.producedMsgChan -> <@HostComms::@Send>(["r1", "p2", "producedMsgChan"]) : !esi.channel<i8>
// CONN:         }
// CONN:         %r1.m1.loopback_fromhw, %r1.p1.producedMsgChan, %r1.p2.producedMsgChan = hw.instance "r1" @Rec(clk: %clk: i1, m1.loopback_tohw: [[r0]]#1: !esi.channel<i8>, c1.consumingFromChan: [[r0]]#2: !esi.channel<i8>) -> (m1.loopback_fromhw: !esi.channel<i8>, p1.producedMsgChan: !esi.channel<i8>, p2.producedMsgChan: !esi.channel<i8>)
// CONN:         hw.output [[r0]]#0 : i8
hw.module @Top2 (%clk: i1) -> (chksum: i8) {
  %c = esi.service.instance svc @HostComms impl as  "topComms2" (%clk) : (i1) -> (i8)
  hw.instance "r1" @Rec(clk: %clk: i1) -> ()
  hw.output %c : i8
}

// CONN-LABEL: hw.module @Rec(%clk: i1, %m1.loopback_tohw: !esi.channel<i8>, %c1.consumingFromChan: !esi.channel<i8>) -> (m1.loopback_fromhw: !esi.channel<i8>, p1.producedMsgChan: !esi.channel<i8>, p2.producedMsgChan: !esi.channel<i8>) {
// CONN:         %m1.loopback_fromhw = hw.instance "m1" @Loopback(clk: %clk: i1, loopback_tohw: %m1.loopback_tohw: !esi.channel<i8>) -> (loopback_fromhw: !esi.channel<i8>)
// CONN:         %c1.rawData = hw.instance "c1" @Consumer(clk: %clk: i1, consumingFromChan: %c1.consumingFromChan: !esi.channel<i8>) -> (rawData: i8)
// CONN:         %p1.producedMsgChan = hw.instance "p1" @Producer(clk: %clk: i1) -> (producedMsgChan: !esi.channel<i8>)
// CONN:         %p2.producedMsgChan = hw.instance "p2" @Producer(clk: %clk: i1) -> (producedMsgChan: !esi.channel<i8>)
// CONN:         hw.output %m1.loopback_fromhw, %p1.producedMsgChan, %p2.producedMsgChan : !esi.channel<i8>, !esi.channel<i8>, !esi.channel<i8>
hw.module @Rec(%clk: i1) -> () {
  hw.instance "m1" @Loopback (clk: %clk: i1) -> ()
  hw.instance "c1" @Consumer(clk: %clk: i1) -> (rawData: i8)
  hw.instance "p1" @Producer(clk: %clk: i1) -> ()
  hw.instance "p2" @Producer(clk: %clk: i1) -> ()
}

// CONN-LABEL: hw.module @Consumer(%clk: i1, %consumingFromChan: !esi.channel<i8>) -> (rawData: i8) {
// CONN:         %true = hw.constant true
// CONN:         %rawOutput, %valid = esi.unwrap.vr %consumingFromChan, %true : i8
// CONN:         hw.output %rawOutput : i8
hw.module @Consumer(%clk: i1) -> (rawData: i8) {
  %dataIn = esi.service.req.to_client <@HostComms::@Recv> (["consumingFromChan"]) : !esi.channel<i8>
  %rdy = hw.constant 1 : i1
  %rawData, %valid = esi.unwrap.vr %dataIn, %rdy: i8
  hw.output %rawData : i8
}

// CONN-LABEL: hw.module @Producer(%clk: i1) -> (producedMsgChan: !esi.channel<i8>) {
// CONN:         %c0_i8 = hw.constant 0 : i8
// CONN:         %true = hw.constant true
// CONN:         %chanOutput, %ready = esi.wrap.vr %c0_i8, %true : i8
// CONN:         hw.output %chanOutput : !esi.channel<i8>
hw.module @Producer(%clk: i1) -> () {
  %data = hw.constant 0 : i8
  %valid = hw.constant 1 : i1
  %dataIn, %rdy = esi.wrap.vr %data, %valid : i8
  esi.service.req.to_server %dataIn -> <@HostComms::@Send> (["producedMsgChan"]) : !esi.channel<i8>
}

// CONN-LABEL: msft.module @MsTop {} (%clk: i1) -> (chksum: i8)
// CONN:         [[r1:%.+]]:2 = esi.service.impl_req svc @HostComms impl as "topComms"(%clk) : (i1) -> (i8, !esi.channel<i8>) {
// CONN:           [[r2:%.+]] = esi.service.req.to_client <@HostComms::@Recv>(["m1", "loopback_tohw"]) : !esi.channel<i8>
// CONN:           esi.service.req.to_server %m1.loopback_fromhw -> <@HostComms::@Send>(["m1", "loopback_fromhw"]) : !esi.channel<i8>
// CONN:         }
// CONN:         %m1.loopback_fromhw = msft.instance @m1 @MsLoopback(%clk, [[r1]]#1) : (i1, !esi.channel<i8>) -> !esi.channel<i8>
// CONN:         msft.output [[r1]]#0 : i8
msft.module @MsTop {} (%clk: i1) -> (chksum: i8) {
  %c = esi.service.instance svc @HostComms impl as  "topComms" (%clk) : (i1) -> (i8)
  msft.instance @m1 @MsLoopback (%clk) : (i1) -> ()
  msft.output %c : i8
}

// CONN-LABEL: msft.module @MsLoopback {} (%clk: i1, %loopback_tohw: !esi.channel<i8>) -> (loopback_fromhw: !esi.channel<i8>)
// CONN:         msft.output %loopback_tohw : !esi.channel<i8>
msft.module @MsLoopback {} (%clk: i1) -> () {
  %dataIn = esi.service.req.to_client <@HostComms::@Recv> (["loopback_tohw"]) : !esi.channel<i8>
  esi.service.req.to_server %dataIn -> <@HostComms::@Send> (["loopback_fromhw"]) : !esi.channel<i8>
  msft.output
}



// CONN-LABEL: msft.module @InOutTop {} (%clk: i1) -> (chksum: i8) {
// CONN:          %0:2 = esi.service.impl_req svc @HostComms impl as "topComms"(%clk) : (i1) -> (i8, !esi.channel<i16>) {
// CONN:            esi.service.req.inout %m1.loopback_inout -> <@HostComms::@ReqResp>(["m1", "loopback_inout"]) : !esi.channel<i8> -> !esi.channel<i16>
// CONN:          }
// CONN:          %m1.loopback_inout = msft.instance @m1 @InOutLoopback(%clk, %0#1)  : (i1, !esi.channel<i16>) -> !esi.channel<i8>
// CONN:          msft.output %0#0 : i8
msft.module @InOutTop {} (%clk: i1) -> (chksum: i8) {
  %c = esi.service.instance svc @HostComms impl as  "topComms" (%clk) : (i1) -> (i8)
  msft.instance @m1 @InOutLoopback (%clk) : (i1) -> ()
  msft.output %c : i8
}

// CONN-LABEL: msft.module @InOutLoopback {} (%clk: i1, %loopback_inout: !esi.channel<i16>) -> (loopback_inout: !esi.channel<i8>) {
// CONN:          %rawOutput, %valid = esi.unwrap.vr %loopback_inout, %ready : i16
// CONN:          %0 = comb.extract %rawOutput from 0 : (i16) -> i8
// CONN:          %chanOutput, %ready = esi.wrap.vr %0, %valid : i8
// CONN:          msft.output %chanOutput : !esi.channel<i8>
msft.module @InOutLoopback {} (%clk: i1) -> () {
  %dataIn = esi.service.req.inout %dataTrunc -> <@HostComms::@ReqResp> (["loopback_inout"]) : !esi.channel<i8> -> !esi.channel<i16>
  %unwrap, %valid = esi.unwrap.vr %dataIn, %rdy: i16
  %trunc = comb.extract %unwrap from 0 : (i16) -> (i8)
  %dataTrunc, %rdy = esi.wrap.vr %trunc, %valid : i8
  msft.output
}

// CONN-LABEL: msft.module @LoopbackCosimTop {} (%clk: i1, %rst: i1) {
// CONN:         [[R1:%.+]] = esi.cosim %clk, %rst, %m1.loopback_inout, "m1.loopback_inout" : !esi.channel<i8> -> !esi.channel<i16>
// CONN:         %m1.loopback_inout = msft.instance @m1 @InOutLoopback(%clk, %0)  : (i1, !esi.channel<i16>) -> !esi.channel<i8>
msft.module @LoopbackCosimTop {} (%clk: i1, %rst: i1) {
  esi.service.instance svc @HostComms impl as "cosim" (%clk, %rst) : (i1, i1) -> ()
  msft.instance @m1 @InOutLoopback(%clk) : (i1) -> ()
  msft.output
}

// CONN-LABEL:  esi.pure_module @LoopbackCosimPure {
// CONN-NEXT:     [[clk:%.+]] = esi.pure_module.input "clk" : i1
// CONN-NEXT:     [[rst:%.+]] = esi.pure_module.input "rst" : i1
// CONN-NEXT:     [[r2:%.+]] = esi.cosim [[clk]], [[rst]], %m1.loopback_inout, "m1.loopback_inout" : !esi.channel<i8> -> !esi.channel<i16>
// CONN-NEXT:     esi.service.hierarchy.metadata path [] implementing @HostComms impl as "cosim" clients [{client_name = ["m1", "loopback_inout"], port = #hw.innerNameRef<@HostComms::@ReqResp>, to_client_type = !esi.channel<i16>, to_server_type = !esi.channel<i8>}]
// CONN-NEXT:     %m1.loopback_inout = msft.instance @m1 @InOutLoopback([[clk]], [[r2]])  : (i1, !esi.channel<i16>) -> !esi.channel<i8>
esi.pure_module @LoopbackCosimPure {
  %clk = esi.pure_module.input "clk" : i1
  %rst = esi.pure_module.input "rst" : i1
  esi.service.instance svc @HostComms impl as "cosim" (%clk, %rst) : (i1, i1) -> ()
  msft.instance @m1 @InOutLoopback(%clk) : (i1) -> ()
}

// CHECK-LABEL: esi.mem.ram @MemA i64 x 20
// CHECK-LABEL: hw.module @MemoryAccess1(%clk: i1, %rst: i1, %write: !esi.channel<!hw.struct<address: i5, data: i64>>, %readAddress: !esi.channel<i5>) -> (readData: !esi.channel<i64>, writeDone: !esi.channel<i0>) {
// CHECK:         esi.service.instance svc @MemA impl as "sv_mem"(%clk, %rst) : (i1, i1) -> ()
// CHECK:         [[DONE:%.+]] = esi.service.req.inout %write -> <@MemA::@write>([]) : !esi.channel<!hw.struct<address: i5, data: i64>> -> !esi.channel<i0>
// CHECK:         [[READ_DATA:%.+]] = esi.service.req.inout %readAddress -> <@MemA::@read>([]) : !esi.channel<i5> -> !esi.channel<i64>
// CHECK:         hw.output [[READ_DATA]], [[DONE]] : !esi.channel<i64>, !esi.channel<i0>

// CONN-LABEL: esi.mem.ram @MemA i64 x 20
// CONN-LABEL: hw.module @MemoryAccess1(%clk: i1, %rst: i1, %write: !esi.channel<!hw.struct<address: i5, data: i64>>, %readAddress: !esi.channel<i5>) -> (readData: !esi.channel<i64>, writeDone: !esi.channel<i0>) {
// CONN:         %MemA = sv.reg  : !hw.inout<uarray<20xi64>>
// CONN:         %chanOutput, %ready = esi.wrap.vr %c0_i0, %write_done : i0
// CONN:         %rawOutput, %valid = esi.unwrap.vr %write, %ready : !hw.struct<address: i5, data: i64>
// CONN:         %address = hw.struct_extract %rawOutput["address"] : !hw.struct<address: i5, data: i64>
// CONN:         %data = hw.struct_extract %rawOutput["data"] : !hw.struct<address: i5, data: i64>
// CONN:         %[[ANDVR:.*]] = comb.and %valid, %ready {sv.namehint = "write_go"} : i1
// CONN:         %write_done = seq.compreg sym @write_done %[[ANDVR]], %clk, %rst, %false  : i1
// CONN:         %chanOutput_0, %ready_1 = esi.wrap.vr %[[MEMREAD:.*]], %valid_3 : i64
// CONN:         %rawOutput_2, %valid_3 = esi.unwrap.vr %readAddress, %ready_1 : i5
// CONN:         %[[MEMREADIO:.*]] = sv.array_index_inout %MemA[%rawOutput_2] : !hw.inout<uarray<20xi64>>, i5
// CONN:         %[[MEMREAD]] = sv.read_inout %[[MEMREADIO]] : !hw.inout<i64>
// CONN:         sv.alwaysff(posedge %clk) {
// CONN:           sv.if %[[ANDVR]] {
// CONN:             %[[ARRIDX:.*]] = sv.array_index_inout %MemA[%address] : !hw.inout<uarray<20xi64>>, i5
// CONN:             sv.passign %[[ARRIDX]], %data : i64
// CONN:           }
// CONN:         }(syncreset : posedge %rst) {
// CONN:         }
// CONN:         hw.output %chanOutput_0, %chanOutput : !esi.channel<i64>, !esi.channel<i0>

esi.mem.ram @MemA i64 x 20
!write = !hw.struct<address: i5, data: i64>
hw.module @MemoryAccess1(%clk: i1, %rst: i1, %write: !esi.channel<!write>, %readAddress: !esi.channel<i5>) -> (readData: !esi.channel<i64>, writeDone: !esi.channel<i0>) {
  esi.service.instance svc @MemA impl as "sv_mem" (%clk, %rst) : (i1, i1) -> ()
  %done = esi.service.req.inout %write -> <@MemA::@write> ([]) : !esi.channel<!write> -> !esi.channel<i0>
  %readData = esi.service.req.inout %readAddress -> <@MemA::@read> ([]) : !esi.channel<i5> -> !esi.channel<i64>
  hw.output %readData, %done : !esi.channel<i64>, !esi.channel<i0>
}

// CONN-LABEL: hw.module @MemoryAccess2Read(%clk: i1, %rst: i1, %write: !esi.channel<!hw.struct<address: i5, data: i64>>, %readAddress: !esi.channel<i5>, %readAddress2: !esi.channel<i5>) -> (readData: !esi.channel<i64>, readData2: !esi.channel<i64>, writeDone: !esi.channel<i0>) {
// CONN:         %MemA = sv.reg : !hw.inout<uarray<20xi64>>
// CONN:         esi.service.hierarchy.metadata path [] implementing @MemA impl as "sv_mem" clients [{client_name = [], port = #hw.innerNameRef<@MemA::@write>, to_client_type = !esi.channel<i0>, to_server_type = !esi.channel<!hw.struct<address: i5, data: i64>>}, {client_name = [], port = #hw.innerNameRef<@MemA::@read>, to_client_type = !esi.channel<i64>, to_server_type = !esi.channel<i5>}, {client_name = [], port = #hw.innerNameRef<@MemA::@read>, to_client_type = !esi.channel<i64>, to_server_type = !esi.channel<i5>}]
// CONN:         hw.output %chanOutput_0, %chanOutput_4, %chanOutput : !esi.channel<i64>, !esi.channel<i64>, !esi.channel<i0>

hw.module @MemoryAccess2Read(%clk: i1, %rst: i1, %write: !esi.channel<!write>, %readAddress: !esi.channel<i5>, %readAddress2: !esi.channel<i5>) -> (readData: !esi.channel<i64>, readData2: !esi.channel<i64>, writeDone: !esi.channel<i0>) {
  esi.service.instance svc @MemA impl as "sv_mem" (%clk, %rst) : (i1, i1) -> ()
  %done = esi.service.req.inout %write -> <@MemA::@write> ([]) : !esi.channel<!write> -> !esi.channel<i0>
  %readData = esi.service.req.inout %readAddress -> <@MemA::@read> ([]) : !esi.channel<i5> -> !esi.channel<i64>
  %readData2 = esi.service.req.inout %readAddress2 -> <@MemA::@read> ([]) : !esi.channel<i5> -> !esi.channel<i64>
  hw.output %readData, %readData2, %done : !esi.channel<i64>, !esi.channel<i64>, !esi.channel<i0>
}

// Check that it doesn't crap out on external modules.
hw.module.extern @extern()

// RUN: circt-opt %s --esi-connect-services -split-input-file -verify-diagnostics

sv.interface @IData {
  sv.interface.signal @data : i32
  sv.interface.signal @valid : i1
  sv.interface.signal @stall : i1
  sv.interface.modport @Sink (output @data, output @valid, input @ready)
}

hw.module @test() {
  %ifaceOut = sv.interface.instance : !sv.interface<@IData>
  %ifaceOutSink = sv.modport.get %ifaceOut @Sink: !sv.interface<@IData> -> !sv.modport<@IData::@Sink>
  // expected-error @+1 {{Interface is not a valid ESI interface.}}
  %idataChanOut = esi.wrap.iface %ifaceOutSink: !sv.modport<@IData::@Sink> -> !esi.channel<i32>
}

// -----

sv.interface @IData {
  sv.interface.signal @data : i2
  sv.interface.signal @valid : i1
  sv.interface.signal @ready : i1
  sv.interface.modport @Sink (output @data, output @valid, input @ready)
}

hw.module @test() {
  %ifaceOut = sv.interface.instance : !sv.interface<@IData>
  %ifaceOutSink = sv.modport.get %ifaceOut @Sink: !sv.interface<@IData> -> !sv.modport<@IData::@Sink>
  // expected-error @+1 {{Operation specifies '!esi.channel<i32>' but type inside doesn't match interface data type 'i2'.}}
  %idataChanOut = esi.wrap.iface %ifaceOutSink: !sv.modport<@IData::@Sink> -> !esi.channel<i32>
}

// -----

sv.interface @IData {
  sv.interface.signal @data : i2
  sv.interface.signal @valid : i1
  sv.interface.signal @ready : i1
  sv.interface.modport @Sink (output @data, output @valid, input @ready)
}

hw.module @test(%m : !sv.modport<@IData::@Noexist>) {
  // expected-error @+1 {{Could not find modport @IData::@Noexist in symbol table.}}
  %idataChanOut = esi.wrap.iface %m: !sv.modport<@IData::@Noexist> -> !esi.channel<i32>
}

// -----

esi.service.decl @HostComms {
  esi.service.to_server @Send : !esi.channel<i16>
  esi.service.to_client @Recv : !esi.channel<i32>
}

hw.module @Loopback (%clk: i1) -> () {
  %dataIn = esi.service.req.to_client <@HostComms::@Recv> (["loopback_tohw"]) : !esi.channel<i32>
  // expected-error @+1 {{'esi.service.req.to_server' op Request to_server type does not match port type '!esi.channel<i16>'}}
  esi.service.req.to_server %dataIn -> <@HostComms::@Send> (["loopback_fromhw"]) : !esi.channel<i32>
}

// -----

esi.service.decl @HostComms {
}

hw.module @Loopback (%clk: i1) -> () {
  // expected-error @+1 {{'esi.service.req.to_client' op Could not locate port "Recv"}}
  %dataIn = esi.service.req.to_client <@HostComms::@Recv> (["loopback_tohw"]) : !esi.channel<i32>
}
// -----

esi.service.decl @HostComms {
  esi.service.to_client @Recv : !esi.channel<i8>
}

hw.module @Loopback (%clk: i1) -> () {
  // expected-error @+1 {{'esi.service.req.to_client' op Request to_client type does not match port type '!esi.channel<i8>'}}
  %dataIn = esi.service.req.to_client <@HostComms::@Recv> (["loopback_tohw"]) : !esi.channel<i32>
}

// -----

esi.mem.ram @MemA i64 x 20
!write = !hw.struct<address: i5, data: i64>
hw.module @MemoryAccess1(%clk: i1, %rst: i1, %write: !esi.channel<!write>) -> () {
  // expected-error @+1 {{'esi.service.instance' op failed to generate server}}
  esi.service.instance svc @MemA impl as "sv_mem" (%clk, %rst) : (i1, i1) -> ()
  // expected-error @+1 {{'esi.service.req.to_server' op Memory write requests must be to/from server}}
  esi.service.req.to_server %write -> <@MemA::@write> ([]) : !esi.channel<!write>
}

// -----

esi.mem.ram @MemA i64 x 20
hw.module @MemoryAccess1(%clk: i1, %rst: i1, %addr: !esi.channel<i5>) -> () {
  // expected-error @+1 {{'esi.service.instance' op failed to generate server}}
  esi.service.instance svc @MemA impl as "sv_mem" (%clk, %rst) : (i1, i1) -> ()
  // expected-error @+1 {{'esi.service.req.to_server' op Memory read requests must be to/from server}}
  esi.service.req.to_server %addr -> <@MemA::@read> ([]) : !esi.channel<i5>
}

// -----

esi.service.decl @HostComms {
  esi.service.inout @ReqResp : !esi.channel<i8> -> !esi.channel<i16>
}

hw.module @Loopback (%clk: i1, %s: !esi.channel<i16>) -> () {
  // expected-error @+1 {{'esi.service.req.inout' op Request to_server type does not match port type '!esi.channel<i8>'}}
  %dataIn = esi.service.req.inout %s -> <@HostComms::@ReqResp> (["loopback_tohw"]) : !esi.channel<i16> -> !esi.channel<i16>
}

// -----

esi.service.decl @HostComms {
  esi.service.inout @ReqResp : !esi.channel<i8> -> !esi.channel<i16>
}

hw.module @Loopback (%clk: i1, %s: !esi.channel<i8>) -> () {
  // expected-error @+1 {{'esi.service.req.inout' op Request to_client type does not match port type '!esi.channel<i16>'}}
  %dataIn = esi.service.req.inout %s -> <@HostComms::@ReqResp> (["loopback_tohw"]) : !esi.channel<i8> -> !esi.channel<i8>
}

// -----

hw.module @Loopback (%clk: i1) -> () {
  // expected-error @+1 {{'esi.service.req.to_client' op Could not find service declaration @HostComms}}
  %dataIn = esi.service.req.to_client <@HostComms::@Recv> (["loopback_tohw"]) : !esi.channel<i32>
}

// -----

esi.service.decl @HostComms {
  esi.service.inout @ReqResp : !esi.channel<i8> -> !esi.channel<i16>
}

hw.module @Top(%clk: i1, %rst: i1) -> () {
  // expected-error @+2 {{'esi.service.impl_req' op did not recognize option name "badOpt"}}
  // expected-error @+1 {{'esi.service.instance' op failed to generate server}}
  esi.service.instance svc @HostComms impl as  "cosim" opts {badOpt = "wrong!"} (%clk, %rst) : (i1, i1) -> ()
}

// -----

!TypeA = !hw.struct<bar: i6>
// expected-error @+1 {{invalid field name: "header5"}}
!TypeAwin1 = !esi.window<
  "TypeAwin1", !TypeA, [
    <"FrameA", [
      <"header5">
    ]>
  ]>

hw.module.extern @TypeAModuleDst(%windowed: !TypeAwin1)

// -----

!TypeA = !hw.struct<bar: i6>
// expected-error @+1 {{cannot specify num items on non-array field "bar"}}
!TypeAwin1 = !esi.window<
  "TypeAwin1", !TypeA, [
    <"FrameA", [
      <"bar", 4>
    ]>
  ]>

hw.module.extern @TypeAModuleDst(%windowed: !TypeAwin1)

// -----

!TypeA = !hw.struct<bar: !hw.array<5xi2>>
// expected-error @+1 {{num items is larger than array size}}
!TypeAwin1 = !esi.window<
  "TypeAwin1", !TypeA, [
    <"FrameA", [
      <"bar", 8>
    ]>
  ]>

hw.module.extern @TypeAModuleDst(%windowed: !TypeAwin1)

// -----

!TypeA = !hw.struct<foo : i3, bar: !hw.array<5xi2>>
// expected-error @+1 {{array with size specified must be in their own frame (in "bar")}}
!TypeAwin1 = !esi.window<
  "TypeAwin1", !TypeA, [
    <"FrameA", [
      <"foo">,
      <"bar", 5>
    ]>
  ]>

hw.module.extern @TypeAModuleDst(%windowed: !TypeAwin1)

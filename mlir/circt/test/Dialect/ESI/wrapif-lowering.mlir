// RUN: circt-opt %s --lower-esi-to-hw -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

sv.interface @IValidReady_i4 {
  sv.interface.signal @valid : i1
  sv.interface.signal @ready : i1
  sv.interface.signal @data : i4
  sv.interface.modport @source  (input @ready, output @valid, output @data)
  sv.interface.modport @sink  (input @valid, input @data, output @ready)
}

hw.module @test(%clk:i1, %rst:i1) {

  %validReady1 = sv.interface.instance : !sv.interface<@IValidReady_i4>
  %0 = sv.modport.get %validReady1 @source : !sv.interface<@IValidReady_i4> -> !sv.modport<@IValidReady_i4::@source>
  %1 = esi.wrap.iface %0 : !sv.modport<@IValidReady_i4::@source> -> !esi.channel<i4>

  %validReady2 = sv.interface.instance : !sv.interface<@IValidReady_i4>
  %2 = sv.modport.get %validReady2 @sink : !sv.interface<@IValidReady_i4> -> !sv.modport<@IValidReady_i4::@sink>
  esi.unwrap.iface %1 into %2 : (!esi.channel<i4>, !sv.modport<@IValidReady_i4::@sink>)

  // CHECK:      %validReady1 = sv.interface.instance  : !sv.interface<@IValidReady_i4>
  // CHECK-NEXT: %[[#modport1:]] = sv.modport.get %validReady1 @source : !sv.interface<@IValidReady_i4> -> !sv.modport<@IValidReady_i4::@source>
  // CHECK-NEXT: %[[#signal1:]] = sv.interface.signal.read %validReady1(@IValidReady_i4::@valid) : i1
  // CHECK-NEXT: %[[#signal2:]] = sv.interface.signal.read %validReady1(@IValidReady_i4::@data) : i4
  // CHECK-NEXT: sv.interface.signal.assign %validReady1(@IValidReady_i4::@ready) = %[[#signal3:]] : i1
  // CHECK-NEXT: %validReady2 = sv.interface.instance  : !sv.interface<@IValidReady_i4>
  // CHECK-NEXT: %[[#modport2:]] = sv.modport.get %validReady2 @sink : !sv.interface<@IValidReady_i4> -> !sv.modport<@IValidReady_i4::@sink>
  // CHECK-NEXT: %[[#signal3:]] = sv.interface.signal.read %validReady2(@IValidReady_i4::@ready) : i1
  // CHECK-NEXT: sv.interface.signal.assign %validReady2(@IValidReady_i4::@valid) = %[[#signal1:]] : i1
  // CHECK-NEXT: sv.interface.signal.assign %validReady2(@IValidReady_i4::@data) = %[[#signal2:]] : i4
}

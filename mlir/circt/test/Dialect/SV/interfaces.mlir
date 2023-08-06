// RUN: circt-opt %s | FileCheck %s
// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: module {
module {

  // Empty interface test
  sv.interface @empty { }

  // CHECK-NEXT: sv.interface @empty {
  // CHECK-NEXT: }

  // Basic interface smoke test

  sv.interface @myinterface {
    sv.interface.signal @data : i32
    sv.interface.modport @input_port (input @data)
    sv.interface.modport @output_port (output @data)
  }

  // CHECK-NEXT: sv.interface @myinterface {
  // CHECK-NEXT:   sv.interface.signal @data : i32
  // CHECK-NEXT:   sv.interface.modport @input_port (input @data)
  // CHECK-NEXT:   sv.interface.modport @output_port (output @data)
  // CHECK-NEXT: }

  // Handshake-like interface smoke test

  sv.interface @handshake_example {
    sv.interface.signal @data : i32
    sv.interface.signal @valid : i1
    sv.interface.signal @ready : i1
    sv.interface.modport @dataflow_in (input @data, input @valid, output @ready)
    sv.interface.modport @dataflow_out (output @data, output @valid, input @ready)
  }

  // CHECK-NEXT: sv.interface @handshake_example {
  // CHECK-NEXT:   sv.interface.signal @data : i32
  // CHECK-NEXT:   sv.interface.signal @valid : i1
  // CHECK-NEXT:   sv.interface.signal @ready : i1
  // CHECK-NEXT:   sv.interface.modport @dataflow_in (input @data, input @valid, output @ready)
  // CHECK-NEXT:   sv.interface.modport @dataflow_out (output @data, output @valid, input @ready)
  // CHECK-NEXT: }

  hw.module.extern @Rcvr (%m: !sv.modport<@handshake_example::@dataflow_in>)
  // CHECK-LABEL: hw.module.extern @Rcvr(%m: !sv.modport<@handshake_example::@dataflow_in>)

  hw.module @Top () {
    %iface = sv.interface.instance : !sv.interface<@handshake_example>
    %ifaceInPort = sv.modport.get %iface @dataflow_in :
      !sv.interface<@handshake_example> -> !sv.modport<@handshake_example::@dataflow_in>
    hw.instance "rcvr" @Rcvr(m: %ifaceInPort: !sv.modport<@handshake_example::@dataflow_in>) -> ()

    %ifaceData = sv.interface.signal.read %iface (@handshake_example::@data) : i32
    %zero32 = arith.constant 0 : i32
    sv.interface.signal.assign %iface(@handshake_example::@data) = %zero32 : i32
  }
  // CHECK-LABEL: hw.module @Top() {
  // CHECK-NEXT:    %iface = sv.interface.instance : !sv.interface<@handshake_example>
  // CHECK-NEXT:    %[[#modport:]] = sv.modport.get %iface @dataflow_in : !sv.interface<@handshake_example> -> !sv.modport<@handshake_example::@dataflow_in>
  // CHECK-NEXT:    hw.instance "rcvr" @Rcvr(m: %[[#modport:]]: !sv.modport<@handshake_example::@dataflow_in>) -> ()
  // CHECK-NEXT:    %[[#]] = sv.interface.signal.read %iface(@handshake_example::@data) : i32
  // CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
  // CHECK-NEXT:    sv.interface.signal.assign %iface(@handshake_example::@data) = %c0_i32 : i32
}

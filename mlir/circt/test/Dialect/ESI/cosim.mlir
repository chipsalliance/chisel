// REQUIRES: capnp
// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s --esi-emit-collateral=schema-file=%t1.capnp --lower-esi-ports --lower-esi-to-hw --export-verilog -verify-diagnostics | FileCheck --check-prefix=COSIM %s

hw.module.extern @Sender() -> (x: !esi.channel<si14>)
hw.module.extern @Reciever(%a: !esi.channel<i32>)
hw.module.extern @ArrReciever(%x: !esi.channel<!hw.array<4xsi64>>)

// CHECK-LABEL: hw.module.extern @Sender() -> (x: !esi.channel<si14>)
// CHECK-LABEL: hw.module.extern @Reciever(%a: !esi.channel<i32>)
// CHECK-LABEL: hw.module.extern @ArrReciever(%x: !esi.channel<!hw.array<4xsi64>>)

hw.module @top(%clk:i1, %rst:i1) -> () {
  hw.instance "recv" @Reciever (a: %cosimRecv: !esi.channel<i32>) -> ()
  // CHECK:  hw.instance "recv" @Reciever(a: %0: !esi.channel<i32>) -> ()

  %send.x = hw.instance "send" @Sender () -> (x: !esi.channel<si14>)
  // CHECK:  %send.x = hw.instance "send" @Sender() -> (x: !esi.channel<si14>)

  %cosimRecv = esi.cosim %clk, %rst, %send.x, "TestEP" : !esi.channel<si14> -> !esi.channel<i32>
  // CHECK:  esi.cosim %clk, %rst, %send.x, "TestEP" : !esi.channel<si14> -> !esi.channel<i32>

  %send2.x = hw.instance "send2" @Sender () -> (x: !esi.channel<si14>)
  // CHECK:  %send2.x = hw.instance "send2" @Sender() -> (x: !esi.channel<si14>)

  %cosimArrRecv = esi.cosim %clk, %rst, %send2.x, "ArrTestEP" : !esi.channel<si14> -> !esi.channel<!hw.array<4xsi64>>
  // CHECK:  esi.cosim %clk, %rst, %send2.x, "ArrTestEP" : !esi.channel<si14> -> !esi.channel<!hw.array<4xsi64>>

  hw.instance "arrRecv" @ArrReciever (x: %cosimArrRecv: !esi.channel<!hw.array<4 x si64>>) -> ()

  // Ensure that the file hash is deterministic.
  // COSIM: @0xccf233b58d85e822;
  // COSIM-LABEL: struct Si14 @0x9bd5e507cce05cc1
  // COSIM:         i @0 :Int16;
  // COSIM-LABEL: struct I32 @0x92cd59dfefaacbdb
  // COSIM:         i @0 :UInt32;
  // Ensure the standard RPC interface is tacked on.
  // COSIM: interface CosimDpiServer
  // COSIM: list @0 () -> (ifaces :List(EsiDpiInterfaceDesc));
  // COSIM: open @1 [S, T] (iface :EsiDpiInterfaceDesc) -> (iface :EsiDpiEndpoint(S, T));

  // COSIM-LABEL: hw.module @top(%clk: i1, %rst: i1)
  // COSIM: %TestEP.DataOutValid, %TestEP.DataOut, %TestEP.DataInReady = hw.instance "TestEP" @Cosim_Endpoint<ENDPOINT_ID_EXT: none = "", SEND_TYPE_ID: ui64 = 11229133067582987457, SEND_TYPE_SIZE_BITS: i32 = 128, RECV_TYPE_ID: ui64 = 10578209918096690139, RECV_TYPE_SIZE_BITS: i32 = 128>(clk: %clk: i1, rst: %rst: i1, DataOutReady: %{{.+}}: i1, DataInValid: %{{.+}}: i1, DataIn: %{{.+}}: !hw.array<128xi1>) -> (DataOutValid: i1, DataOut: !hw.array<128xi1>, DataInReady: i1)
}

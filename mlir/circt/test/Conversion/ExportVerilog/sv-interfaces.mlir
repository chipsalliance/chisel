// RUN: circt-opt %s -export-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

module {
  // CHECK-LABEL: interface data_vr;
  // CHECK:         logic [31:0] data;
  // CHECK:         logic valid;
  // CHECK:         logic ready;
  // CHECK:         logic [3:0][7:0] arrayData;
  // CHECK:         logic [7:0] uarrayData[0:3];
  // CHECK:         modport data_in(input data, input valid, output ready);
  // CHECK:         modport data_out(output data, output valid, input ready);
  // CHECK:         MACRO(data, valid, ready -- data_in)
  // CHECK:         // logic /*Zero Width*/ zeroGround;
  // CHECK:         // logic [3:0]/*Zero Width*/ zeroArray;
  // CHECK:         // logic /*Zero Width*/ zeroUArray[0:3];
  // CHECK:       endinterface
  // CHECK-EMPTY:
  sv.interface @data_vr {
    sv.interface.signal @data : i32
    sv.interface.signal @valid : i1
    sv.interface.signal @ready : i1
    sv.interface.signal @arrayData : !hw.array<4xi8>
    sv.interface.signal @uarrayData : !hw.uarray<4xi8>
    sv.interface.modport @data_in (input @data, input @valid, output @ready)
    sv.interface.modport @data_out (output @data, output @valid, input @ready)
    sv.verbatim  "//MACRO({{0}}, {{1}}, {{2}} -- {{3}})"
                    {symbols = [@data, @valid, @ready, @data_in]}
    sv.interface.signal @zeroGround : i0
    sv.interface.signal @zeroArray : !hw.array<4xi0>
    sv.interface.signal @zeroUArray : !hw.uarray<4xi0>
  }

  // CHECK-LABEL: interface struct_vr;
  // CHECK:         struct packed {logic [6:0] foo; logic [4:0][15:0] bar; } data;
  // CHECK:         logic valid;
  // CHECK:         logic ready;
  // CHECK:         modport data_in(input data, input valid, output ready);
  // CHECK:         modport data_out(output data, output valid, input ready);
  // CHECK:       endinterface
  // CHECK-EMPTY:
  sv.interface @struct_vr {
    sv.interface.signal @data : !hw.struct<foo: i7, bar: !hw.array<5 x i16>>
    sv.interface.signal @valid : i1
    sv.interface.signal @ready : i1
    sv.interface.modport @data_in (input @data, input @valid, output @ready)
    sv.interface.modport @data_out (output @data, output @valid, input @ready)
  }

  hw.module.extern @Rcvr (%m: !sv.modport<@data_vr::@data_in>)

  // CHECK-LABEL: module Top
  hw.module @Top (%clk: i1) {
    // CHECK: data_vr [[IFACE:.+]]();
    %iface = sv.interface.instance : !sv.interface<@data_vr>
    // CHECK: MACRO-Interface:data_vr
    sv.verbatim "//MACRO-Interface:{{0}}" {symbols = [@data_vr]}
    // CHECK: struct_vr [[IFACEST:.+]]();
    %structIface = sv.interface.instance : !sv.interface<@struct_vr>

    %ifaceInPort = sv.modport.get %iface @data_in :
      !sv.interface<@data_vr> -> !sv.modport<@data_vr::@data_in>

    // CHECK: Rcvr rcvr1 ({{.*}}//{{.+}}
    // CHECK:   .m ([[IFACE]].data_in){{.*}}//{{.+}}
    // CHECK: );
    hw.instance "rcvr1" @Rcvr(m: %ifaceInPort: !sv.modport<@data_vr::@data_in>) -> ()

    // CHECK: Rcvr rcvr2 ({{.*}}//{{.+}}
    // CHECK:   .m ([[IFACE]].data_in){{.*}}//{{.+}}
    // CHECK: );
    hw.instance "rcvr2" @Rcvr(m: %ifaceInPort: !sv.modport<@data_vr::@data_in>) -> ()

    %c1 = hw.constant 1 : i1
    // CHECK: assign iface.valid = 1'h1;
    sv.interface.signal.assign %iface(@data_vr::@valid) = %c1 : i1

    sv.always posedge %clk {
      %fd = hw.constant 0x80000002 : i32

      %validValue = sv.interface.signal.read %iface(@data_vr::@valid) : i1
      // CHECK: $fwrite(32'h80000002, "valid: %d\n", iface.valid);
      sv.fwrite %fd, "valid: %d\n" (%validValue) : i1
      // CHECK: assert(iface.valid);
      sv.assert %validValue, immediate

      sv.if %clk {
        %structDataSignal = sv.interface.signal.read %structIface(@struct_vr::@data) : !hw.struct<foo: i7, bar: !hw.array<5 x i16>>
        %structData = hw.struct_extract %structDataSignal["foo"] : !hw.struct<foo: i7, bar: !hw.array<5 x i16>>
        // CHECK: $fwrite(32'h80000002, "%d", [[IFACEST]].data.foo);
        sv.fwrite %fd, "%d"(%structData) : i7
      }
    }
  }

// Next test case is related to:https://github.com/llvm/circt/issues/681
// Rename keywords used in variable/module names
// NOTE(fschuiki): Extern modules should trigger an error diagnostic if they
// would cause a rename, but since the user supplies the module externally we
// can't just rename it.
  hw.module.extern @regStuff (%m: !sv.modport<@data_vr::@data_in>)
  // CHECK-LABEL: module Top2
  hw.module @Top2 (%clk: i1) {
    // CHECK: data_vr [[IFACE:.+]]();{{.*}}//{{.+}}
    %iface = sv.interface.instance : !sv.interface<@data_vr>

    %ifaceInPort = sv.modport.get %iface @data_in :
      !sv.interface<@data_vr> -> !sv.modport<@data_vr::@data_in>

    // CHECK: regStuff rcvr1 ({{.*}}//{{.+}}
    // CHECK:   .m ([[IFACE]].data_in){{.*}}//{{.+}}
    // CHECK: );
    hw.instance "rcvr1" @regStuff(m: %ifaceInPort: !sv.modport<@data_vr::@data_in>) -> ()

    // CHECK: regStuff rcvr2 ({{.*}}//{{.+}}
    // CHECK:   .m ([[IFACE]].data_in){{.*}}//{{.+}}
    // CHECK: );
    hw.instance "rcvr2" @regStuff(m: %ifaceInPort: !sv.modport<@data_vr::@data_in>) -> ()
  }

  // https://github.com/llvm/circt/issues/724
  sv.interface @IValidReady_Struct  {
    sv.interface.signal @data : !hw.struct<foo: !hw.array<384xi1>>
  }
  // CHECK-LABEL: module structs(
  // CHECK-NOT: wire [383:0] _tmp =
  // CHECK: wire struct packed {logic [383:0] foo; } _GEN
  // CHECK: endmodule
  hw.module @structs(%clk: i1, %rstn: i1) {
    %0 = sv.interface.instance name "iface" : !sv.interface<@IValidReady_Struct>
    sv.interface.signal.assign %0(@IValidReady_Struct::@data) = %s : !hw.struct<foo: !hw.array<384xi1>>
    %c0 = hw.constant 0 : i8
    %c64 = hw.constant 100000 : i64
    %16 = hw.bitcast %c64 : (i64) -> !hw.array<64xi1>
    %58 = hw.bitcast %c0 : (i8) -> !hw.array<8xi1>
    %90 = hw.array_concat %58, %58, %58, %58, %58, %58, %58, %58, %58, %58, %58, %58, %58, %58, %58, %58, %58, %58, %58, %58, %58, %58, %58, %58, %58, %58, %58, %58, %58, %58, %58, %58, %16, %16 : !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<8xi1>, !hw.array<64xi1>, !hw.array<64xi1>
    %s = hw.struct_create (%90) : !hw.struct<foo: !hw.array<384xi1>>
  }

  // CHECK-LABEL: interface renameType;
  sv.interface @renameType {
    sv.interface.signal @data : !hw.struct<repeat: i1>
    // CHECK-NEXT: struct packed {logic repeat_0; } data;
  }

  // CHECK-LABEL: // interface with a comment
  // CHECK-NEXT:  interface interfaceWithComment
  sv.interface @interfaceWithComment
    attributes {comment = "interface with a comment"} {}
}

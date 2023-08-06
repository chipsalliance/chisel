// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-mem-to-reg-of-vec))' %s | FileCheck  %s

firrtl.circuit "Mem" attributes {annotations = [{class = "sifive.enterprise.firrtl.ConvertMemToRegOfVecAnnotation$"}]}{
  firrtl.module public @Mem(out %d : !firrtl.probe<vector<uint<8>, 8>>, out %d2 : !firrtl.probe<vector<uint<8>, 8>>) attributes {annotations = [
    {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}
  ]} {
    %dbg, %mem_read, %mem_write, %debug = firrtl.mem Undefined {
      depth = 8 : i64,
      name = "mem",
      portNames = ["dbg", "read", "write", "debug"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32
    } : !firrtl.probe<vector<uint<8>, 8>>, !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>,
        !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>,
        !firrtl.probe<vector<uint<8>, 8>>
    firrtl.ref.define %d, %debug : !firrtl.probe<vector<uint<8>, 8>>
    firrtl.ref.define %d2, %dbg : !firrtl.probe<vector<uint<8>, 8>>
  }
    // CHECK-LABEL: firrtl.circuit "Mem" {
    // CHECK:         firrtl.module public @Mem(
    // CHECK:           %mem_read = firrtl.wire  : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>
    // CHECK:           %[[v0:.+]] = firrtl.subfield %mem_read[addr]
    // CHECK:           %[[v1:.+]] = firrtl.subfield %mem_read[en]
    // CHECK:           %[[v2:.+]] = firrtl.subfield %mem_read[clk]
    // CHECK:           %[[v3:.+]] = firrtl.subfield %mem_read[data]
    // CHECK:           %mem = firrtl.reg %[[v6:.+]]  : !firrtl.clock, !firrtl.vector<uint<8>, 8>
    // CHECK:           %[[v23:.+]] = firrtl.subaccess %mem[%[[v4:.+]]]
    // CHECK:           %invalid_ui8 = firrtl.invalidvalue : !firrtl.uint<8>
    // CHECK:           firrtl.strictconnect %[[v3]], %invalid_ui8 : !firrtl.uint<8>
    // CHECK:           firrtl.when %[[v1]] : !firrtl.uint<1> {
    // CHECK:             firrtl.strictconnect %[[v3]], %[[v23]]
    // CHECK:           }
    // CHECK:           %mem_write = firrtl.wire  : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    // CHECK:           %[[v5:.+]] = firrtl.subfield %mem_write[addr]
    // CHECK:           %[[v6:.+]] = firrtl.subfield %mem_write[en]
    // CHECK:           %[[v7:.+]] = firrtl.subfield %mem_write[clk]
    // CHECK:           %[[v8:.+]] = firrtl.subfield %mem_write[data]
    // CHECK:           %[[v9:.+]] = firrtl.subfield %mem_write[mask]
    // CHECK:           %[[v10:.+]] = firrtl.subaccess %mem[%[[v5]]]
    // CHECK:           firrtl.when %[[v6]] : !firrtl.uint<1> {
    // CHECK:             firrtl.when %[[v9]] : !firrtl.uint<1> {
    // CHECK:               firrtl.strictconnect %[[v10]], %[[v8]] : !firrtl.uint<8>
    // CHECK:             }
    // CHECK:           }
    // CHECK:           %11 = firrtl.ref.send %mem : !firrtl.vector<uint<8>, 8>
    // CHECK:           %12 = firrtl.ref.send %mem : !firrtl.vector<uint<8>, 8>
    // CHECK:           firrtl.ref.define %d, %12 : !firrtl.probe<vector<uint<8>, 8>>
    // CHECK:           firrtl.ref.define %d2, %11 : !firrtl.probe<vector<uint<8>, 8>>


}

firrtl.circuit "Mem_Ignore" {
  firrtl.module public @Mem_Ignore() attributes {annotations = [
    {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}
  ]} {
    %mem_read, %mem_write = firrtl.mem Undefined {
      depth = 8 : i64,
      name = "mem",
      portNames = ["read", "write"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>,
        !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    // CHECK:      %mem_read, %mem_write = firrtl.mem Undefined
    // CHECK-SAME:   {depth = 8 : i64, name = "mem", portNames = ["read", "write"], readLatency = 0 : i32, writeLatency = 1 : i32}
    // CHECK-SAME:   : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>,
    // CHECK-SAME:     !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
  }
}

firrtl.circuit  "GCTModule" attributes {annotations = [
  {class = "sifive.enterprise.firrtl.ConvertMemToRegOfVecAnnotation$"}
]} {
  firrtl.module public @GCTModule() attributes {annotations = [
    {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}
  ]} {
    %rf_read, %rf_write = firrtl.mem Undefined {
      annotations = [
        {circt.fieldID = 1 : i64, class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 1 : i64, type = "source"},
        {circt.fieldID = 1 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 2 : i64, class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 2 : i64, type = "source"},
        {circt.fieldID = 2 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 3 : i64, class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 3 : i64, type = "source"},
        {circt.fieldID = 3 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 4 : i64, class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 4 : i64, type = "source"},
        {circt.fieldID = 4 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 5 : i64, class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 5 : i64, type = "source"},
        {circt.fieldID = 5 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 6 : i64, class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 6 : i64, type = "source"},
        {circt.fieldID = 6 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 7 : i64, class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 7 : i64, type = "source"},
        {circt.fieldID = 7 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 8 : i64, class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 8 : i64, type = "source"},
        {circt.fieldID = 8 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 1 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 2 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 3 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 4 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 5 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 6 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 7 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 8 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 1 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 2 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 3 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 4 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 5 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 6 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 7 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 8 : i64, class = "firrtl.transforms.DontTouchAnnotation"}
      ],
      depth = 8 : i64,
      name = "rf",
      portNames = ["read",
      "write"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>,
        !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
      // CHECK-LABEL: firrtl.module public @GCTModule()
      // CHECK:         %rf = firrtl.reg %2
      // CHECK-SAME:      {circt.fieldID = 1 : i64, class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 1 : i64, type = "source"},
      // CHECK-SAME:      {circt.fieldID = 1 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
      // CHECK-SAME:      {circt.fieldID = 2 : i64, class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 2 : i64, type = "source"},
      // CHECK-SAME:      {circt.fieldID = 2 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
      // CHECK-SAME:      {circt.fieldID = 3 : i64, class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 3 : i64, type = "source"},
      // CHECK-SAME:      {circt.fieldID = 3 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
      // CHECK-SAME:      {circt.fieldID = 4 : i64, class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 4 : i64, type = "source"},
      // CHECK-SAME:      {circt.fieldID = 4 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
      // CHECK-SAME:      {circt.fieldID = 5 : i64, class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 5 : i64, type = "source"},
      // CHECK-SAME:      {circt.fieldID = 5 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
      // CHECK-SAME:      {circt.fieldID = 6 : i64, class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 6 : i64, type = "source"},
      // CHECK-SAME:      {circt.fieldID = 6 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
      // CHECK-SAME:      {circt.fieldID = 7 : i64, class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 7 : i64, type = "source"},
      // CHECK-SAME:      {circt.fieldID = 7 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
      // CHECK-SAME:      {circt.fieldID = 8 : i64, class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 8 : i64, type = "source"},
      // CHECK-SAME:      {circt.fieldID = 8 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
      // CHECK-SAME:      {circt.fieldID = 1 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
      // CHECK-SAME:      {circt.fieldID = 2 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
      // CHECK-SAME:      {circt.fieldID = 3 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
      // CHECK-SAME:      {circt.fieldID = 4 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
      // CHECK-SAME:      {circt.fieldID = 5 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
      // CHECK-SAME:      {circt.fieldID = 6 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
      // CHECK-SAME:      {circt.fieldID = 7 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
      // CHECK-SAME:      {circt.fieldID = 8 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
      // CHECK-SAME:      {circt.fieldID = 1 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
      // CHECK-SAME:      {circt.fieldID = 2 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
      // CHECK-SAME:      {circt.fieldID = 3 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
      // CHECK-SAME:      {circt.fieldID = 4 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
      // CHECK-SAME:      {circt.fieldID = 5 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
      // CHECK-SAME:      {circt.fieldID = 6 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
      // CHECK-SAME:      {circt.fieldID = 7 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
      // CHECK-SAME:      {circt.fieldID = 8 : i64, class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.clock, !firrtl.vector<uint<8>, 8>
  }
}

firrtl.circuit "WriteMask" attributes {annotations = [
  {class = "sifive.enterprise.firrtl.ConvertMemToRegOfVecAnnotation$"}
]} {
  firrtl.module public @WriteMask() attributes {annotations = [
    {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}
  ]} {
    %mem_read, %mem_write = firrtl.mem Undefined {
      depth = 8 : i64,
      name = "mem",
      portNames = ["read", "write"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: vector<uint<8>, 2>>,
        !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: vector<uint<8>, 2>, mask: vector<uint<1>, 2>>
    // CHECK-LABEL: firrtl.module public @WriteMask()
    // CHECK:         %mem = firrtl.reg %2  : !firrtl.clock, !firrtl.vector<vector<uint<8>, 2>, 8>
    // CHECK:         %mem_write = firrtl.wire  : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: vector<uint<8>, 2>, mask: vector<uint<1>, 2>>
    // CHECK:         %[[v5:.+]] = firrtl.subfield %mem_write[addr]
    // CHECK:         %[[v6:.+]] = firrtl.subfield %mem_write[en]
    // CHECK:         %[[v7:.+]] = firrtl.subfield %mem_write[clk]
    // CHECK:         %[[v8:.+]] = firrtl.subfield %mem_write[data]
    // CHECK:         %[[v9:.+]] = firrtl.subfield %mem_write[mask]
    // CHECK:         %[[v10:.+]] = firrtl.subaccess %mem[%5] : !firrtl.vector<vector<uint<8>, 2>, 8>, !firrtl.uint<3>
    // CHECK:         %[[v11:.+]] = firrtl.subindex
    // CHECK:         %[[v12:.+]] = firrtl.subindex
    // CHECK:         %[[v13:.+]] = firrtl.subindex
    // CHECK:         %[[v14:.+]] = firrtl.subindex
    // CHECK:         %[[v15:.+]] = firrtl.subindex
    // CHECK:         %[[v16:.+]] = firrtl.subindex
    // CHECK:         firrtl.when %[[v6]] : !firrtl.uint<1> {
    // CHECK:           firrtl.when %[[v13]] : !firrtl.uint<1> {
    // CHECK:             firrtl.strictconnect %[[v11]], %[[v12]] : !firrtl.uint<8>
    // CHECK:           }
    // CHECK:           firrtl.when %[[v16]] : !firrtl.uint<1> {
    // CHECK:             firrtl.strictconnect %[[v14]], %[[v15]] : !firrtl.uint<8>
    // CHECK:           }
    %mem_read1, %mem_write1 = firrtl.mem Undefined {
      annotations = [
        {class = "sifive.enterprise.firrtl.ExcludeMemFromMemToRegOfVec"}
      ],
      depth = 8 : i64,
      name = "mem",
      portNames = ["read", "write"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: vector<uint<8>, 2>>,
        !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: vector<uint<8>, 2>, mask: vector<uint<1>, 2>>
    // CHECK: %mem_read_0, %mem_write_1 = firrtl.mem Undefined {depth = 8 :
  }
}

firrtl.circuit "MemTap" attributes {annotations = [
  {class = "sifive.enterprise.firrtl.ConvertMemToRegOfVecAnnotation$"}
]} {
  firrtl.module public @MemTap() attributes {annotations = [
    {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}
  ]} {
		%rf_MPORT, %rf_io_rdata_0_MPORT, %rf_io_rdata_1_MPORT = firrtl.mem sym @rf Undefined {
      annotations = [
        {class = "sifive.enterprise.grandcentral.MemTapAnnotation.source", id = 11 : i64}
      ],
      depth = 4 : i64,
      name = "rf",
      portNames = ["MPORT", "io_rdata_0_MPORT", "io_rdata_1_MPORT"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data: uint<32>, mask: uint<1>>,
        !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data flip: uint<32>>,
        !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data flip: uint<32>>
    // CHECK-LABEL: firrtl.module public @MemTap()
    // CHECK:         %rf = firrtl.reg sym @rf %2
    // CHECK-SAME:      [{circt.fieldID = 1 : i64, class = "sifive.enterprise.grandcentral.MemTapAnnotation.source", id = 11 : i64, portID = 0 : i64}
    // CHECK-SAME:       {circt.fieldID = 2 : i64, class = "sifive.enterprise.grandcentral.MemTapAnnotation.source", id = 11 : i64, portID = 1 : i64},
    // CHECK-SAME:       {circt.fieldID = 3 : i64, class = "sifive.enterprise.grandcentral.MemTapAnnotation.source", id = 11 : i64, portID = 2 : i64},
    // CHECK-SAME:       {circt.fieldID = 4 : i64, class = "sifive.enterprise.grandcentral.MemTapAnnotation.source", id = 11 : i64, portID = 3 : i64}]}
    // CHECK-SAME:      : !firrtl.clock, !firrtl.vector<uint<32>, 4>
	}

}

// Test the behavior of non-local annotations using either the old or new
// format work correctly.
//
// CHECK-LABEL: "NLA"
firrtl.circuit "NLA" attributes {annotations = [
  {class = "sifive.enterprise.firrtl.ConvertMemToRegOfVecAnnotation$"}
]} {
  // The hierachical paths are unchanged.
  // CHECK:      hw.hierpath private @path_old [@NLA::@foo, @Foo::@old]
  // CHECK-NEXT: hw.hierpath private @path_new [@NLA::@foo, @Foo]
  hw.hierpath private @path_old [@NLA::@foo, @Foo::@old]
  hw.hierpath private @path_new [@NLA::@foo, @Foo]
  firrtl.module private @Foo() {
    // CHECK:      %old = firrtl.reg sym @old
    // CHECK-SAME:   {circt.nonlocal = @path, class = "oldNLA"}
    %old_r = firrtl.mem sym @old Undefined {
      annotations = [
        {circt.nonlocal = @path, class = "oldNLA"}
      ],
      depth = 4 : i64,
      name = "old",
      portNames = ["r"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data flip: uint<32>>
    // CHECK:      %new = firrtl.reg
    // CHECK-NOT:    sym
    // CHECK-SAME:   {circt.nonlocal = @path, class = "newNLA"}
    %new_r = firrtl.mem Undefined {
      annotations = [
        {circt.nonlocal = @path, class = "newNLA"}
      ],
      depth = 4 : i64,
      name = "new",
      portNames = ["r"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data flip: uint<32>>
  }
  firrtl.module public @NLA() attributes {annotations = [
    {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}
  ]} {
    firrtl.instance foo sym @foo @Foo()
	}
}

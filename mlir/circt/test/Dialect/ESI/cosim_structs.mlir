// REQUIRES: capnp
// RUN: circt-translate %s -export-esi-capnp -verify-diagnostics | FileCheck --check-prefix=CAPNP %s
// RUN: circt-opt %s --lower-esi-ports --lower-esi-to-hw -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck --check-prefix=COSIM %s

!DataPkt = !hw.struct<encrypted: i1, compressionLevel: ui4, blob: !hw.array<32 x i8>>
!pktChan = !esi.channel<!DataPkt>

hw.module.extern @Compressor(%in: !esi.channel<i1>) -> (x: !pktChan)

hw.module @top(%clk:i1, %rst:i1) -> () {
  %compressedData = hw.instance "compressor" @Compressor(in: %inputData: !esi.channel<i1>) -> (x: !pktChan)
  %inputData = esi.cosim %clk, %rst, %compressedData, "Compressor" : !pktChan -> !esi.channel<i1>
}

// CAPNP:      struct Struct{{.+}}
// CAPNP-NEXT:   encrypted        @0 :Bool;
// CAPNP-NEXT:   compressionLevel @1 :UInt8;
// CAPNP-NEXT:   blob             @2 :List(UInt8);

// COSIM-DAG: hw.instance "encodeStruct{{.+}}Inst" @encodeStruct{{.+}}(clk: %clk: i1, valid: %{{.+}}: i1, unencodedInput: %{{.+}}: !hw.struct<encrypted: i1, compressionLevel: ui4, blob: !hw.array<32xi8>>) -> (encoded: !hw.array<448xi1>)
// COSIM-DAG: hw.instance "Compressor" @Cosim_Endpoint<ENDPOINT_ID_EXT: none = "", SEND_TYPE_ID: ui64 = 11116741711825659895, SEND_TYPE_SIZE_BITS: i32 = 448, RECV_TYPE_ID: ui64 = 17519082812652290511, RECV_TYPE_SIZE_BITS: i32 = 128>(clk: %clk: i1, rst: %rst: i1, {{.+}}, {{.+}}, {{.+}}) -> ({{.+}})
// COSIM-DAG: hw.module @encode{{.+}}(%clk: i1, %valid: i1, %unencodedInput: !hw.struct<encrypted: i1, compressionLevel: ui4, blob: !hw.array<32xi8>>) -> (encoded: !hw.array<448xi1>)

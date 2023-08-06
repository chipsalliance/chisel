// REQUIRES: esi-cosim
// RUN: circt-opt %s --lower-esi-to-physical --lower-esi-ports --lower-esi-to-hw | circt-opt --export-verilog -o %t3.mlir > %t1.sv
// RUN: circt-translate %s -export-esi-capnp -verify-diagnostics > %t2.capnp
// RUN: esi-cosim-runner.py --schema %t2.capnp %s %t1.sv %S/../supplements/integers.sv
// PY: import basic
// PY: rpc = basic.BasicSystemTester(rpcschemapath, simhostport)
// PY: print(rpc.list())
// PY: rpc.testIntAcc(25)
// PY: rpc.testVectorSum(25)
// PY: rpc.testCrypto(25)

hw.module.extern @IntAccNoBP(%clk: i1, %rst: i1, %ints: !esi.channel<i32>) -> (totalOut: !esi.channel<i32>) attributes {esi.bundle}
hw.module.extern @IntArrSum(%clk: i1, %rst: i1, %arr: !esi.channel<!hw.array<4 x si13>>) -> (totalOut: !esi.channel<!hw.array<2 x ui24>>) attributes {esi.bundle}

hw.module @ints(%clk: i1, %rst: i1) {
  %intsIn = esi.cosim %clk, %rst, %intsTotalBuffered, "TestEP" : !esi.channel<i32> -> !esi.channel<i32>
  %intsInBuffered = esi.buffer %clk, %rst, %intsIn {stages=2, name="intChan"} : i32
  %intsTotal = hw.instance "acc" @IntAccNoBP(clk: %clk: i1, rst: %rst: i1, ints: %intsInBuffered: !esi.channel<i32>) -> (totalOut: !esi.channel<i32>)
  %intsTotalBuffered = esi.buffer %clk, %rst, %intsTotal {stages=2, name="totalChan"} : i32
}

hw.module @array(%clk: i1, %rst: i1) {
  %arrIn = esi.cosim %clk, %rst, %arrTotalBuffered, "TestEP" : !esi.channel<!hw.array<2 x ui24>> -> !esi.channel<!hw.array<4 x si13>>
  %arrInBuffered = esi.buffer %clk, %rst, %arrIn {stages=2, name="arrChan"} : !hw.array<4 x si13>
  %arrTotal = hw.instance "acc" @IntArrSum(clk: %clk: i1, rst: %rst: i1, arr: %arrInBuffered: !esi.channel<!hw.array<4 x si13>>) -> (totalOut: !esi.channel<!hw.array<2 x ui24>>)
  %arrTotalBuffered = esi.buffer %clk, %rst, %arrTotal {stages=2, name="totalChan"} : !hw.array<2 x ui24>
}

!DataPkt = !hw.struct<encrypted: i1, blob: !hw.array<32 x i8>>
!pktChan = !esi.channel<!DataPkt>
!Config  = !hw.struct<encrypt:   i1, otp:  !hw.array<32 x i8>>
!cfgChan = !esi.channel<!Config>

hw.module.extern @Encryptor(%clk: i1, %rst: i1, %in: !pktChan, %cfg: !cfgChan) -> (x: !pktChan) attributes {esi.bundle}

hw.module @structs(%clk:i1, %rst:i1) -> () {
  %compressedData = hw.instance "otpCryptor" @Encryptor(clk: %clk: i1, rst: %rst: i1, in: %inputData: !pktChan, cfg: %cfg: !cfgChan) -> (x: !pktChan)
  %inputData = esi.cosim %clk, %rst, %compressedData, "CryptoData" : !pktChan -> !pktChan
  %c0 = hw.constant 0 : i1
  %null, %nullReady = esi.wrap.vr %c0, %c0 : i1
  %cfg = esi.cosim %clk, %rst, %null, "CryptoConfig" : !esi.channel<i1> -> !cfgChan
}

hw.module @top(%clk: i1, %rst: i1) {
  hw.instance "ints" @ints (clk: %clk: i1, rst: %rst: i1) -> ()
  hw.instance "array" @array(clk: %clk: i1, rst: %rst: i1) -> ()
  hw.instance "structs" @structs(clk: %clk: i1, rst: %rst: i1) -> ()
}

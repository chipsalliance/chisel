// REQUIRES: esi-cosim
// RUN: rm -rf %t6 && mkdir %t6 && cd %t6
// RUN: circt-opt %s --esi-connect-services --esi-emit-collateral=schema-file=%t2.capnp --esi-clean-metadata > %t4.mlir
// RUN: circt-opt %t4.mlir --lower-esi-to-physical --lower-esi-ports --lower-esi-to-hw --export-split-verilog -o %t3.mlir
// RUN: circt-translate %t4.mlir -export-esi-capnp -verify-diagnostics > %t2.capnp
// RUN: cd ..
// RUN: esi-cosim-runner.py --schema %t2.capnp --exec %S/loopback.py %t6/*.sv


hw.module @intLoopback(%clk:i1, %rst:i1) -> () {
  %cosimRecv = esi.cosim %clk, %rst, %bufferedResp, "IntTestEP" {name_ext="loopback"} : !esi.channel<i32> -> !esi.channel<i32>
  %bufferedResp = esi.buffer %clk, %rst, %cosimRecv {stages=1} : i32
}

!KeyText = !hw.struct<text: !hw.array<6xi14>, key: !hw.array<4xi8>>
hw.module @twoListLoopback(%clk:i1, %rst:i1) -> () {
  %cosim = esi.cosim %clk, %rst, %resp, "KeyTextEP" : !esi.channel<!KeyText> -> !esi.channel<!KeyText>
  %resp = esi.buffer %clk, %rst, %cosim {stages=4} : !KeyText
}

esi.service.decl @HostComms {
  esi.service.to_server @Send : !esi.channel<!esi.any>
  esi.service.to_client @Recv : !esi.channel<i8>
}

hw.module @TwoChanLoopback(%clk: i1) -> () {
  %dataIn = esi.service.req.to_client <@HostComms::@Recv> (["loopback_tohw"]) : !esi.channel<i8>
  esi.service.req.to_server %dataIn -> <@HostComms::@Send> (["loopback_fromhw"]) : !esi.channel<i8>
}

hw.module @top(%clk:i1, %rst:i1) -> () {
  hw.instance "intLoopbackInst" @intLoopback(clk: %clk: i1, rst: %rst: i1) -> ()
  hw.instance "twoListLoopbackInst" @twoListLoopback(clk: %clk: i1, rst: %rst: i1) -> ()

  esi.service.instance svc @HostComms impl as  "cosim" (%clk, %rst) : (i1, i1) -> ()
  hw.instance "TwoChanLoopback" @TwoChanLoopback(clk: %clk: i1) -> ()
}

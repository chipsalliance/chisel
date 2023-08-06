// REQUIRES: esi-cosim
// RUN: rm -rf %t && mkdir %t && cd %t
// RUN: circt-opt %s --esi-connect-services --esi-emit-cpp-api="output-file=ESISystem.h" --esi-emit-collateral=schema-file=%t/schema.capnp --esi-clean-metadata > %t/4.mlir
// RUN: circt-opt %t/4.mlir --lower-esi-to-physical --lower-esi-ports --lower-esi-to-hw --export-split-verilog -o %t/3.mlir
// RUN: circt-translate %t/4.mlir -export-esi-capnp -verify-diagnostics > %t/schema.capnp

// Build the project using the CMakeLists.txt from this directory. Just move
// everything to the output folder in the build directory; this is very convenient
// if we want to run the build manually afterwards.
// RUN: cp %S/loopback.cpp %t/
// RUN: cp %S/CMakeLists.txt %t/
// RUN: cmake -S %t \
// RUN:   -B %t/build \
// RUN:   -DCIRCT_DIR=%CIRCT_SOURCE% \
// RUN:   -DCAPNP_SCHEMA=%t/schema.capnp
// RUN: cmake --build %t/build

// Run test
// RUN: esi-cosim-runner.py --tmpdir=%t   \
// RUN:     --schema %t/hw/schema.capnp   \
// RUN:     --exec %t/build/loopback_test \
// RUN:     $(ls %t/encode*.sv) $(ls %t/decode*.sv) \
// RUN:     %t/intLoopback.sv %t/twoListLoopback.sv %t/TwoChanLoopback.sv \
// RUN:     %t/top.sv

// To run this test manually:
// 1. run `ninja check-circt-integration` (this will create the output folder, run the initial passes, ...)
// 2. navigate to %t
// 3. In a separate terminal, run esi-cosim-runner.py in server only mode:
//   - cd %t
//   - esi-cosim-runner.py --tmpdir=$(pwd) --schema=$(pwd)/hw/schema.capnp --server-only %{the .sv files in the right order}
// 4. In another terminal, run the test executable (%t/build/loopback_test). When running esi-cosim-runner, it'll print the $port which
//    the test executable should connect to.
//   - cd %t/build
//   - ./loopback_test localhost:$port ../hw/schema.capn


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
  esi.service.to_server @Send : !esi.channel<i8>
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

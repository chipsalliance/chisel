// RUN: circt-translate -split-input-file --export-calyx --verify-diagnostics %s

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%in0: i4, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i8, %done: i1 {done}) {
    %true = hw.constant true
    // expected-error @+2 {{'calyx.std_extsi' op not supported for emission}}
    // expected-note @+1 {{calyx.std_extsi is currently not available in the native Rust compiler (see github.com/cucapra/calyx/issues/1009)}}
    %std_extsi.in, %std_extsi.out = calyx.std_extsi @std_extsi : i4, i8
    calyx.wires {
      calyx.assign %std_extsi.in = %in0 : i4
      calyx.assign %out0 = %std_extsi.out : i8
      calyx.assign %done = %true : i1
    }
    calyx.control {}
  }
}


// RUN: circt-opt %s --split-input-file -handshake-lock-functions --verify-diagnostics

// expected-error @+1 {{cannot lock a region without arguments}}
handshake.func @no_arg() -> none {
  %ctrl = source
  return %ctrl : none
}

// -----

// expected-error @+1 {{cannot lock a region without results}}
handshake.func @no_res(%ctrl: none) {
  sink %ctrl : none
  return
}

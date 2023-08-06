// RUN: circt-opt --split-input-file -allow-unregistered-dialect --hw-eliminate-inout-ports -verify-diagnostics %s

hw.module @unsupported(%a: !hw.inout<i42>) {
  // expected-error @+1 {{uses hw.inout port "a" but the operation itself is unsupported.}}
  "foo.bar" (%a) : (!hw.inout<i42>) -> ()
}

// -----

// expected-error @+1 {{multiple writers of inout port "a" is unsupported.}}
hw.module @multipleWriters(%a: !hw.inout<i42>) {
  %0 = hw.constant 0 : i42
  sv.assign %a, %0 : i42
  sv.assign %a, %0 : i42
}

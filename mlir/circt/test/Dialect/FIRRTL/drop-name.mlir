// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-drop-names{preserve-values=all})))' %s   | FileCheck %s --check-prefix=ALL
// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-drop-names{preserve-values=named})))' %s | FileCheck %s --check-prefix=NAMED
// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-drop-names{preserve-values=none})))' %s  | FileCheck %s --check-prefix=NONE

firrtl.circuit "Foo" {
  firrtl.module @Foo() {
    // ALL:   %a = firrtl.wire  interesting_name : !firrtl.uint<1>
    // NAMED: %a = firrtl.wire  interesting_name : !firrtl.uint<1>
    // NONE:  %a = firrtl.wire  : !firrtl.uint<1>
    %a = firrtl.wire interesting_name : !firrtl.uint<1>

    // ALL:   %_a = firrtl.wire  interesting_name : !firrtl.uint<1>
    // NAMED: %_a = firrtl.wire  : !firrtl.uint<1>
    // NONE:  %_a = firrtl.wire  : !firrtl.uint<1>
    %_a = firrtl.wire interesting_name : !firrtl.uint<1>

    // ALL:   %_T = firrtl.wire : !firrtl.uint<1>
    // NAMED: %0 = firrtl.wire  : !firrtl.uint<1>
    // NONE:  %0 = firrtl.wire  : !firrtl.uint<1>
    %_T = firrtl.wire interesting_name : !firrtl.uint<1>

  }
}

// RUN: circt-opt %s | FileCheck %s

firrtl.circuit "Classes" {
  firrtl.module @Classes() {}

  // CHECK-LABEL: firrtl.class @StringPassThru(in %in_str: !firrtl.string, out %out_str: !firrtl.string)
  firrtl.class @StringPassThru(in %in_str: !firrtl.string, out %out_str: !firrtl.string) {
    // CHECK: firrtl.propassign %out_str, %in_str : !firrtl.string
    firrtl.propassign %out_str, %in_str : !firrtl.string
  }

  // CHECK-LABEL: firrtl.module @ModuleWithObjectPort(in %in: !firrtl.class<@StringPassThru(in in_str: !firrtl.string, out out_str: !firrtl.string)>) 
  firrtl.module @ModuleWithObjectPort(in %in: !firrtl.class<@StringPassThru(in in_str: !firrtl.string, out out_str: !firrtl.string)>) {}

  // CHECK-LABEL: firrtl.class @EmptyClass()
  firrtl.class @EmptyClass() {}

  // CHECK-LABEL: firrtl.module @ModuleWithOutputObject(out %out: !firrtl.class<@EmptyClass()>)
  firrtl.module @ModuleWithOutputObject(out %out: !firrtl.class<@EmptyClass()>) {
    %0 = firrtl.object @EmptyClass()
    firrtl.propassign %out, %0 : !firrtl.class<@EmptyClass()>
  }
}

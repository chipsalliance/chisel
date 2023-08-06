// RUN: circt-translate %s --export-systemc --verify-diagnostics --split-input-file | FileCheck %s

// CHECK: <<UNSUPPORTED OPERATION (hw.module)>>
// expected-error @+1 {{no emission pattern found for 'hw.module'}}
hw.module @notSupported () -> () { }

// -----

// CHECK: <<UNSUPPORTED TYPE (!hw.inout<i2>)>>
// expected-error @+1 {{no emission pattern found for type '!hw.inout<i2>'}}
systemc.module @invalidType (%port0: !systemc.in<!hw.inout<i2>>) {}

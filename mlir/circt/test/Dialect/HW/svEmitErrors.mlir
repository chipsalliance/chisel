// RUN: circt-opt -export-verilog -verify-diagnostics --split-input-file %s

// expected-error @+1 {{value has an unsupported verilog type 'vector<3xi1>'}}
hw.module @A(%a: vector<3 x i1>) -> () { }

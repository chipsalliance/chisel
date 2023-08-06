// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s --lower-msft-to-hw --lower-seq-to-sv --msft-export-tcl=tops=shallow,deeper,regions,reg | FileCheck %s --check-prefix=LOWER
// RUN: circt-opt %s --lower-msft-to-hw --lower-seq-to-sv --msft-export-tcl=tops=shallow,deeper,regions,reg --export-verilog | FileCheck %s --check-prefix=TCL

hw.globalRef @ref1 [#hw.innerNameRef<@deeper::@branch>, #hw.innerNameRef<@shallow::@leaf>, #hw.innerNameRef<@leaf::@module>]
msft.pd.location @ref1 M20K x: 15 y: 9 n: 3 path: "|memBank2"

hw.globalRef @ref2 [#hw.innerNameRef<@shallow::@leaf>, #hw.innerNameRef<@leaf::@module>]
msft.pd.location @ref2 M20K x: 8 y: 19 n: 1 path: "|memBank2"

hw.globalRef @ref3 [#hw.innerNameRef<@regions::@module>]
msft.pd.physregion @ref3 @region1 path: "baz"

hw.globalRef @ref4 [#hw.innerNameRef<@reg::@reg>]
msft.pd.location @ref4 FF x: 0 y: 0 n: 0

hw.module.extern @Foo()

// CHECK: msft.entity.extern @entity1 "some tag"
msft.entity.extern @entity1 "some tag"

// CHECK: msft.entity.extern @entity2 {name = "foo", number = 1 : i64}
msft.entity.extern @entity2 {name = "foo", number = 1 : i64}

// CHECK-LABEL: msft.module @leaf
// LOWER-LABEL: hw.module @leaf
msft.module @leaf {} () -> () {
  // CHECK: msft.instance @module @Foo()
  // LOWER: hw.instance "module" sym @module @Foo() -> ()
  // LOWER-NOT: #msft.switch.inst
  msft.instance @module @Foo() {
    circt.globalRef = [#hw.globalNameRef<@ref1>, #hw.globalNameRef<@ref2>]
  } : () -> ()
  // LOWER{LITERAL}: sv.verbatim "proc {{0}}_config
  // LOWER{LITERAL}: sv.verbatim "proc {{0}}_config
  // LOWER{LITERAL}: sv.verbatim "proc {{0}}_config
  msft.output
}

// TCL: Foo module_0

// TCL-NOT: proc leaf_config

// TCL-LABEL: proc shallow_config
msft.module @shallow {} () -> () {
  msft.instance @leaf @leaf() { circt.globalRef = [#hw.globalNameRef<@ref1>, #hw.globalNameRef<@ref2>] } : () -> ()
  // TCL: set_location_assignment M20K_X8_Y19_N1 -to $parent|leaf|module_0|memBank2
  msft.output
}

// TCL-LABEL: proc deeper_config
msft.module @deeper {} () -> () {
  msft.instance @branch @shallow() { circt.globalRef = [#hw.globalNameRef<@ref1>] } : () -> ()
  msft.instance @leaf @leaf() : () -> ()
  // TCL: set_location_assignment M20K_X15_Y9_N3 -to $parent|branch|leaf|module_0|memBank2
  msft.output
}

msft.physical_region @region1, [
  #msft.physical_bounds<x: [0, 10], y: [0, 10]>,
  #msft.physical_bounds<x: [20, 30], y: [20, 30]>]

// TCL-LABEL: proc regions_config
msft.module @regions {} () -> () {
  msft.instance @module @Foo() {
    circt.globalRef = [#hw.globalNameRef<@ref3>]
  } : () -> ()
  // TCL: set_instance_assignment -name PLACE_REGION "X0 Y0 X10 Y10;X20 Y20 X30 Y30" -to $parent|module_0
  // TCL: set_instance_assignment -name RESERVE_PLACE_REGION OFF -to $parent|module_0
  // TCL: set_instance_assignment -name CORE_ONLY_PLACE_REGION ON -to $parent|module_0
  // TCL: set_instance_assignment -name REGION_NAME region1 -to $parent|module_0
  msft.output
}

// TCL-LABEL: proc reg_0_config
msft.module @reg {} (%input : i8, %clk : i1) -> () {
  %reg = seq.compreg sym @reg %input, %clk { circt.globalRef = [#hw.globalNameRef<@ref4>] } : i8
  // TCL: set_location_assignment FF_X0_Y0_N0 -to $parent|reg_0
  msft.output
}

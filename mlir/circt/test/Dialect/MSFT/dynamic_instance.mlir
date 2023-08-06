// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics
// RUN: circt-opt %s --msft-lower-instances --verify-each | FileCheck %s
// RUN: circt-opt %s --msft-lower-instances --lower-msft-to-hw --lower-seq-to-sv --msft-export-tcl=tops=shallow,deeper,reg --export-verilog -o %t.mlir | FileCheck %s --check-prefix=TCL

msft.instance.hierarchy @deeper {
  msft.instance.dynamic @deeper::@branch {
    msft.instance.dynamic @shallow::@leaf {
      msft.instance.dynamic @leaf::@module {
        msft.pd.location M20K x: 15 y: 9 n: 3 path: "|memBank2"
        msft.instance.verb_attr name: "RESERVE_PLACE_REGION" value: "OFF" path: "|memBank2"
      }
    }
  }
}
// CHECK: hw.globalRef @instref [#hw.innerNameRef<@deeper::@branch>, #hw.innerNameRef<@shallow::@leaf>, #hw.innerNameRef<@leaf::@module>]
// CHECK: msft.pd.location @instref M20K x: 15 y: 9 n: 3 path : "|memBank2"

msft.instance.hierarchy @shallow {
  msft.instance.dynamic @shallow::@leaf {
    msft.instance.dynamic @leaf::@module {
      msft.pd.location M20K x: 8 y: 19 n: 1 path: "|memBank2"
    }
  }
}
// CHECK: hw.globalRef @instref_1 [#hw.innerNameRef<@shallow::@leaf>, #hw.innerNameRef<@leaf::@module>]
// CHECK: msft.pd.location @instref_1 M20K x: 8 y: 19 n: 1 path : "|memBank2"

msft.instance.hierarchy @reg "foo" {
  msft.instance.dynamic @reg::@reg {
    msft.pd.reg_location i4 [*, <1,2,3>, <1,2,4>, <1,2,5>]
  }
}
msft.instance.hierarchy @reg "bar" {
  msft.instance.dynamic @reg::@reg {
    msft.pd.reg_location i4 [<3,4,5>, *, *, *]
  }
}
// CHECK: hw.globalRef @instref_2 [#hw.innerNameRef<@reg::@reg>]
// CHECK-DAG: msft.pd.reg_location ref @instref_2 i4 [*, <1, 2, 3>, <1, 2, 4>, <1, 2, 5>]
// CHECK: hw.globalRef @instref_3 [#hw.innerNameRef<@reg::@reg>]
// CHECK-DAG: msft.pd.reg_location ref @instref_3 i4 [<3, 4, 5>, *, *, *]



hw.module.extern @Foo()

// CHECK-LABEL: msft.module @leaf
// LOWER-LABEL: hw.module @leaf
msft.module @leaf {} () -> () {
  // CHECK: msft.instance @module @Foo()
  // LOWER: hw.instance "module" sym @module @Foo() -> ()
  // LOWER-NOT: #msft.switch.inst
  msft.instance @module @Foo() : () -> ()
  // LOWER{LITERAL}: sv.verbatim "proc {{0}}_config
  // LOWER{LITERAL}: sv.verbatim "proc {{0}}_config
  // LOWER{LITERAL}: sv.verbatim "proc {{0}}_config
  msft.output
}

// TCL: Foo module_0

// TCL-NOT: proc leaf_config

// TCL-LABEL: proc shallow_config
msft.module @shallow {} () -> () {
  msft.instance @leaf @leaf() : () -> ()
  // TCL: set_location_assignment M20K_X8_Y19_N1 -to $parent|leaf|module_0|memBank2
  msft.output
}

// TCL-LABEL: proc deeper_config
msft.module @deeper {} () -> () {
  msft.instance @branch @shallow()  : () -> ()
  msft.instance @leaf @leaf() : () -> ()
  // TCL: set_location_assignment M20K_X15_Y9_N3 -to $parent|branch|leaf|module_0|memBank2
  // TCL: set_instance_assignment -name RESERVE_PLACE_REGION OFF -to $parent|branch|leaf|module_0|memBank2

  msft.output
}

msft.module @reg {} (%input : i4, %clk : i1) -> () {
  %reg = seq.compreg sym @reg %input, %clk  : i4
  msft.output
}
// TCL-LABEL: proc reg_0_foo_config
// TCL: set_location_assignment FF_X1_Y2_N3 -to $parent|reg_0[1]
// TCL: set_location_assignment FF_X1_Y2_N4 -to $parent|reg_0[2]
// TCL: set_location_assignment FF_X1_Y2_N5 -to $parent|reg_0[3]

// TCL-LABEL: proc reg_0_bar_config
// TCL: set_location_assignment FF_X3_Y4_N5 -to $parent|reg_0[0]

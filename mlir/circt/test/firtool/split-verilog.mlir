// RUN: firtool %s --format=mlir -verilog | FileCheck %s --check-prefix=VERILOG
// RUN: rm -rf %t
// RUN: firtool %s --format=mlir -split-verilog -o=%t
// RUN: FileCheck %s --check-prefix=VERILOG-FOO < %t/foo.sv
// RUN: FileCheck %s --check-prefix=VERILOG-BAR < %t/bar.sv
// RUN: FileCheck %s --check-prefix=VERILOG-USB < %t/usb.sv
// RUN: FileCheck %s --check-prefix=VERILOG-INOUT-3 < %t/inout_3.sv
// RUN: FileCheck %s --check-prefix=VERILOG-CUSTOM-1 < %t/custom1.sv
// RUN: FileCheck %s --check-prefix=VERILOG-CUSTOM-2 < %t/custom2.sv
// RUN: FileCheck %s --check-prefix=LIST < %t/filelist.f

sv.verbatim "// I'm everywhere"
sv.ifdef "VERILATOR" {
  sv.verbatim "// Hello"
} else {
  sv.verbatim "// World"
}
sv.verbatim ""

hw.type_scope @__hw_typedecls {
  hw.typedecl @foo : i1
}

hw.module @foo(%a: i1) -> (b: i1) {
  hw.output %a : i1
}
hw.module @bar(%x: i1) -> (y: i1) {
  hw.output %x : i1
}
sv.interface @usb {
  sv.interface.signal @valid : i1
  sv.interface.signal @ready : i1
}
hw.module.extern @pll ()

hw.module @inout(%inout: i1) -> (output: i1) {
  hw.output %inout : i1
}

// This is made to collide with the first renaming attempt of the `@inout`
// module above.
hw.module.extern @inout_0 () -> ()
hw.module.extern @inout_1 () -> ()
hw.module.extern @inout_2 () -> ()

sv.verbatim "// Foo" {output_file = #hw.output_file<"custom1.sv">}
sv.verbatim "// Bar" {output_file = #hw.output_file<"custom2.sv", excludeFromFileList>}

// LIST:      {{^}}foo.sv{{$}}
// LIST-NEXT: {{^}}bar.sv{{$}}
// LIST-NEXT: {{^}}usb.sv{{$}}
// LIST-NEXT: {{^}}extern_modules.sv{{$}}
// LIST-NEXT: {{^}}inout_3.sv{{$}}
// LIST-NEXT: {{^}}custom1.sv{{$}}
// LIST-NOT:  custom2.sv

// VERILOG-FOO:       // I'm everywhere
// VERILOG-FOO-NEXT:  `ifdef VERILATOR
// VERILOG-FOO-NEXT:    // Hello
// VERILOG-FOO-NEXT:  `else
// VERILOG-FOO-NEXT:    // World
// VERILOG-FOO-NEXT:  `endif
// VERILOG-FOO:       typedef logic foo;
// VERILOG-FOO-LABEL: module foo(
// VERILOG-FOO:       endmodule

// VERILOG-BAR:       // I'm everywhere
// VERILOG-BAR-NEXT:  `ifdef VERILATOR
// VERILOG-BAR-NEXT:    // Hello
// VERILOG-BAR-NEXT:  `else
// VERILOG-BAR-NEXT:    // World
// VERILOG-BAR-NEXT:  `endif
// VERILOG-BAR:       typedef logic foo;
// VERILOG-BAR-LABEL: module bar
// VERILOG-BAR:       endmodule

// VERILOG-USB:       // I'm everywhere
// VERILOG-USB-NEXT:  `ifdef VERILATOR
// VERILOG-USB-NEXT:    // Hello
// VERILOG-USB-NEXT:  `else
// VERILOG-USB-NEXT:    // World
// VERILOG-USB-NEXT:  `endif
// VERILOG-USB:       typedef logic foo;
// VERILOG-USB-LABEL: interface usb;
// VERILOG-USB:       endinterface

// VERILOG-INOUT-3:       // I'm everywhere
// VERILOG-INOUT-3-NEXT:  `ifdef VERILATOR
// VERILOG-INOUT-3-NEXT:    // Hello
// VERILOG-INOUT-3-NEXT:  `else
// VERILOG-INOUT-3-NEXT:    // World
// VERILOG-INOUT-3-NEXT:  `endif
// VERILOG-INOUT-3:       typedef logic foo;
// VERILOG-INOUT-3-LABEL: module inout_3(
// VERILOG-INOUT-3:       endmodule

// VERILOG-CUSTOM-1: // Foo
// VERILOG-CUSTOM-2: // Bar

// VERILOG:       // I'm everywhere
// VERILOG-NEXT:  `ifdef VERILATOR
// VERILOG-NEXT:    // Hello
// VERILOG-NEXT:  `else
// VERILOG-NEXT:    // World
// VERILOG-NEXT:  `endif
// VERILOG:       typedef logic foo;
// VERILOG-LABEL: module foo(
// VERILOG:       endmodule
// VERILOG-LABEL: module bar
// VERILOG:       endmodule
// VERILOG-LABEL: interface usb;
// VERILOG:       endinterface
// VERILOG:       // external module pll
// VERILOG:       // external module inout_0
// VERILOG:       // external module inout_1
// VERILOG:       // external module inout_2
// VERILOG-LABEL: FILE "custom1.sv"
// VERILOG:       // Foo
// VERILOG-LABEL: FILE "custom2.sv"
// VERILOG:       // Bar

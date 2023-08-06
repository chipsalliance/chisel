// RUN: circt-opt %s | FileCheck %s
// RUN: circt-opt %s | circt-opt | FileCheck %s
// RUN: circt-opt %s --export-verilog -o %t.mlir | FileCheck %s --check-prefix=SV

hw.module @PrintPath<> () -> () {
  %fd = hw.constant 0x80000002 : i32
  sv.initial {
    sv.fwrite %fd, "%m\n"
  }
}

hw.module @Case1<NUM : i8> () -> () {
  %fd = hw.constant 0x80000002 : i32
  sv.generate "foo_case": {
    sv.generate.case #hw.param.decl.ref<"NUM"> : i8 [
      case (0, "case0") {
        sv.initial {
          sv.fwrite %fd, "case 0\n"
        }
      }
      case (1 : i64, "case1") {
        hw.instance "print1" @PrintPath() -> ()
      }
      case (unit, "dflt") {
        hw.instance "printDflt" @PrintPath() -> ()
      }
    ]
  }
}

// CHECK-LABEL: hw.module @Case1
// CHECK-NEXT:    [[FD:%.+]] = hw.constant -2147483646 : i32
// CHECK-NEXT:    sv.generate "foo_case" : {
// CHECK-NEXT:      sv.generate.case #hw.param.decl.ref<"NUM"> : i8 [
// CHECK-NEXT:        case (0 : i64, "case0") {
// CHECK-NEXT:          sv.initial {
// CHECK-NEXT:            sv.fwrite [[FD]], "case 0\0A"
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        case (1 : i64, "case1") {
// CHECK-NEXT:          hw.instance "print1" @PrintPath() -> ()
// CHECK-NEXT:        }
// CHECK-NEXT:        case (unit, "dflt") {
// CHECK-NEXT:          hw.instance "printDflt" @PrintPath() -> ()
// CHECK-NEXT:        }
// CHECK-NEXT:        ]
// CHECK-NEXT:    }

// SV-LABEL: module Case1
// SV:         #(parameter [7:0] NUM) ();
// SV:         generate
// SV:         begin: foo_case
// SV:           case (NUM)
// SV:             64'd0: begin: case0
// SV:               initial
// SV:                 $fwrite(32'h80000002, "case 0\n");
// SV:             end: case0
// SV:             64'd1: begin: case1
// SV:               PrintPath print1 ();
// SV:             end: case1
// SV:             default: begin: dflt
// SV:               PrintPath printDflt ();
// SV:             end: dflt
// SV:           endcase
// SV:         end: foo_case
// SV:         endgenerate
// SV:       endmodule

hw.module @CaseNoDefault<NUM : i8> () -> () {
  sv.generate "bar": {
    sv.generate.case #hw.param.decl.ref<"NUM"> : i8 [
      case (0 : i64, "bar0") { }
    ]
  }
}

// CHECK-LABEL: hw.module @CaseNoDefault
// CHECK:         sv.generate "bar" : {
// CHECK:           sv.generate.case #hw.param.decl.ref<"NUM"> : i8 [
// CHECK:             case (0 : i64, "bar0") {
// CHECK:             }
// CHECK:           ]

// SV-LABEL: module CaseNoDefault
// SV:         #(parameter [7:0] NUM) ();
// SV:         generate
// SV:         begin: bar
// SV:             case (NUM)
// SV:               64'd0: begin: bar0
// SV:               end: bar0
// SV:             endcase
// SV:         end: bar
// SV:         endgenerate
// SV:       endmodule

// RUN: firtool %s --format=mlir --ir-fir | circt-opt | FileCheck %s --check-prefix=MLIR
// RUN: firtool %s --format=mlir --ir-fir --emit-bytecode | circt-opt | FileCheck %s --check-prefix=MLIR
// RUN: circt-opt %s --emit-bytecode | firtool --ir-fir | circt-opt | FileCheck %s --check-prefix=MLIR
// RUN: firtool %s --format=mlir -verilog | FileCheck %s --check-prefix=VERILOG
// RUN: firtool %s --format=mlir -verilog -omir-file %S/firtool.fir.omir.anno.json -output-omir meta.omir.json -output-final-mlir=%t | FileCheck %s --check-prefix=VERILOG-WITH-MLIR
// RUN: firtool %s --format=mlir -verilog -output-final-mlir=%t.mlirbc -emit-bytecode | FileCheck %s --check-prefix=VERILOG-WITH-MLIR
// RUN: FileCheck %s --input-file=%t --check-prefix=VERILOG-WITH-MLIR-OUT
// RUN: circt-opt %t.mlirbc | FileCheck %s --check-prefix=VERILOG-WITH-MLIR-OUT
// RUN: not diff %t %t.mlirbc

firrtl.circuit "Top" {
  firrtl.module @Top(in %in : !firrtl.uint<8>,
                     out %out : !firrtl.uint<8>) {
    firrtl.connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
}

// MLIR-LABEL: firrtl.module @Top(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
// MLIR-NEXT:    firrtl.strictconnect %out, %in : !firrtl.uint<8>
// MLIR-NEXT:  }

// VERILOG-LABEL: module Top(
// VERILOG-NEXT :   input  [7:0] in,
// VERILOG-NEXT :   output [7:0] out);
// VERILOG-NEXT :   assign out = in;
// VERILOG-NEXT : endmodule

// VERILOG-WITH-MLIR-LABEL: module Top(
// VERILOG-WITH-MLIR-NEXT :   input  [7:0] in,
// VERILOG-WITH-MLIR-NEXT :   output [7:0] out);
// VERILOG-WITH-MLIR-NEXT :   assign out = in;
// VERILOG-WITH-MLIR-NEXT : endmodule

// VERILOG-WITH-MLIR-OUT-NOT: sv.verbatim{{.*}}output_file = {{.*}}meta.omir.json

// VERILOG-WITH-MLIR-OUT-LABEL: hw.module @Top(%in: i8) -> (out: i8) {
// VERILOG-WITH-MLIR-OUT-NEXT:    hw.output %in : i8
// VERILOG-WITH-MLIR-OUT-NEXT:  }

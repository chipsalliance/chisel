; Ensure firtool can dump MLIR from various points of the pipeline.
; Check passes from different components and registration categories.
; RUN: firtool %s -mlir-print-ir-before=cse 2>&1 | FileCheck %s -Dpass=CSE
; RUN: firtool %s -mlir-print-ir-before=firrtl-inliner 2>&1 | FileCheck %s -Dpass=Inliner
; RUN: firtool %s -mlir-print-ir-before=firrtl-lower-annotations 2>&1 | FileCheck %s -Dpass=LowerFIRRTLAnnotations
; RUN: firtool %s -mlir-print-ir-before=firrtl-grand-central 2>&1 | FileCheck %s -Dpass=GrandCentral
; RUN: firtool %s -mlir-print-ir-before=export-verilog 2>&1 | FileCheck %s -Dpass=ExportVerilog
; RUN: firtool %s -mlir-print-ir-before=export-split-verilog -split-verilog -o %t 2>&1 | FileCheck %s -Dpass=ExportSplitVerilog
; RUN: firtool %s -mlir-print-ir-before=hw-cleanup 2>&1 | FileCheck %s -Dpass=HWCleanup
; CHECK: IR Dump Before [[pass]]
circuit Empty:
  module Empty:

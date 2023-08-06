// RUN: firtool --help | FileCheck %s --implicit-check-not='{{[Oo]}}ptions:'

// CHECK: OVERVIEW: MLIR-based FIRRTL compiler
// CHECK: General {{[Oo]}}ptions
// CHECK: Generic Options
// CHECK: firtool Options
// CHECK: --lowering-options=

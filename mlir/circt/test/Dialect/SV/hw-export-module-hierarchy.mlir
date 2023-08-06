// RUN: circt-opt -pass-pipeline='builtin.module(hw-export-module-hierarchy)' %s | FileCheck %s

hw.module @InnerModule() -> () {}

hw.module @MainDesign() -> () {
  hw.instance "inner" @InnerModule() -> ()
}

hw.module @TestHarness() attributes {firrtl.moduleHierarchyFile = [#hw.output_file<"testharness_hier.json", excludeFromFileList>]} {
  hw.instance "main_design" @MainDesign() -> ()
}

// CHECK:      hw.module @MainDesign()
// CHECK-NEXT:   hw.instance "inner"
// CHECK-SAME:     sym @[[MainDesign_inner_sym:[_a-zA-Z0-9]+]]

// CHECK:      hw.module @TestHarness()
// CHECK-NEXT:   hw.instance "main_design"
// CHECK-SAME:     sym @[[TestHarness_main_design_sym:[_a-zA-Z0-9]+]]

// CHECK:               sv.verbatim "{
// CHECK-SAME{LITERAL}:   \22instance_name\22: \22{{0}}\22,
// CHECK-SAME{LITERAL}:   \22module_name\22: \22{{0}}\22,
// CHECK-SAME:            \22instances\22: [
// CHECK-SAME:              {
// CHECK-SAME{LITERAL}:       \22instance_name\22: \22{{1}}\22,
// CHECK-SAME{LITERAL}:       \22module_name\22: \22{{2}}\22,
// CHECK-SAME:                \22instances\22: [
// CHECK-SAME:                  {
// CHECK-SAME{LITERAL}:           \22instance_name\22: \22{{3}}\22,
// CHECK-SAME{LITERAL}:           \22module_name\22: \22{{4}}\22,
// CHECK-SAME:                    \22instances\22: []
// CHECK-SAME:                  }
// CHECK-SAME:                ]
// CHECK-SAME:              }
// CHECK-SAME:            ]
// CHECK-SAME:          }"
// CHECK-SAME:         symbols = [@TestHarness,
// CHECK-SAME:           #hw.innerNameRef<@TestHarness::@[[TestHarness_main_design_sym]]>,
// CHECK-SAME:           @MainDesign,
// CHECK-SAME:           #hw.innerNameRef<@MainDesign::@[[MainDesign_inner_sym]]>,
// CHECK-SAME:           @InnerModule]

// RUN: circt-opt %s --verify-diagnostics -pass-pipeline='builtin.module(om-link-modules)' --split-input-file | FileCheck %s

module {
  // CHECK-LABEL: module
  // CHECK-NOT:   module
  // CHECK-NOT:   om.class.extern
  // CHECK-LABEL: om.class @A
  // CHECK-LABEL: om.class @Conflict_A
  // CHECK-LABEL: om.class @UseConflict_A
  // CHECK-NEXT:    om.object @Conflict_A() : () -> !om.class.type<@Conflict_A>
  // CHECK-NEXT:    om.class.field @c, %{{.+}} : !om.class.type<@Conflict_A>

  // CHECK-LABEL: om.class @Conflict_B
  // CHECK-LABEL: om.class @UseConflict_B()
  // CHECK-NEXT:    om.object @Conflict_B() : () -> !om.class.type<@Conflict_B>
  // CHECK-NEXT:    om.object.field %{{.+}}, [@c] : (!om.class.type<@Conflict_B>) -> i1

  // CHECK-LABEL: om.class @Conflict_module_0()
  // CHECK-LABEL: om.class @UseConflict_module_0()
  // CHECK-NEXT:    om.object @Conflict_module_0() : () -> !om.class.type<@Conflict_module_0>
  // CHECK-NEXT:    om.object.field %{{.+}}, [@c] : (!om.class.type<@Conflict_module_0>) -> i1

  module attributes {om.namespace = "A"} {
    om.class @A(%arg: i1) {
      om.class.field @a, %arg: i1
    }
    om.class @Conflict() {
      %0 = om.constant 0 : i1
      om.class.field @c, %0: i1
    }
    om.class @UseConflict() {
      %0 = om.object @Conflict() : () -> !om.class.type<@Conflict>
      om.class.field @c, %0: !om.class.type<@Conflict>
    }
  }
  module attributes {om.namespace = "B"} {
    om.class.extern @A(%arg: i1) {
      om.class.extern.field @a: i1
    }
    om.class @Conflict() {
      %0 = om.constant 0 : i1 
      om.class.field @c, %0: i1
    }
    om.class @UseConflict() {
     %0 = om.object @Conflict() : () -> !om.class.type<@Conflict>
     %1 = om.object.field %0, [@c] : (!om.class.type<@Conflict>) -> i1
     om.class.field @c, %1: i1
    }
  }
  module {
    om.class @Conflict() {
      %0 = om.constant 0 : i1 
      om.class.field @c, %0: i1
    }
    om.class @UseConflict() {
     %0 = om.object @Conflict() : () -> !om.class.type<@Conflict>
     %1 = om.object.field %0, [@c] : (!om.class.type<@Conflict>) -> i1
    }
  }
}

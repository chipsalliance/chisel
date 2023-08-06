// RUN: circt-opt %s | circt-opt | FileCheck %s
// RUN: circt-opt %s -ssp-roundtrip | circt-opt | FileCheck %s --check-prefix=INFRA

// 1) tests the plain parser/printer roundtrip.
// CHECK: ssp.library @Lib {
// CHECK:   operator_type @Opr [latency<1>, limit<1>]
// CHECK: }
// CHECK: module @SomeModule {
// CHECK:   ssp.library @Lib {
// CHECK:     operator_type @Opr [latency<2>, limit<2>]
// CHECK:   }
// CHECK: }
// CHECK: ssp.instance @SomeInstance of "ModuloProblem" {
// CHECK:   library @InternalLib {
// CHECK:     operator_type @Opr [latency<3>, limit<3>]
// CHECK:   }
// CHECK:   graph {
// CHECK:     operation<@Opr>()
// CHECK:     operation<@InternalLib::@Opr>()
// CHECK:     operation<@SomeInstance::@InternalLib::@Opr>()
// CHECK:     operation<@Lib::@Opr>()
// CHECK:     operation<@SomeModule::@Lib::@Opr>()
// CHECK:   }
// CHECK: }

// 2) Import/export via the scheduling infra (i.e. populates a `Problem` instance and reconstructs the SSP IR from it.)
//    Operator types from stand-alone libraries are appended to the instance's internal library.
// INFRA: ssp.instance @SomeInstance of "ModuloProblem" {
// INFRA:   library @InternalLib {
// INFRA:     operator_type @Opr [latency<3>, limit<3>]
// INFRA:     operator_type @Opr_1 [latency<1>, limit<1>]
// INFRA:     operator_type @Opr_2 [latency<2>, limit<2>]
// INFRA:   }
// INFRA:   graph {
// INFRA:     operation<@Opr>()
// INFRA:     operation<@Opr>()
// INFRA:     operation<@Opr>()
// INFRA:     operation<@Opr_1>()
// INFRA:     operation<@Opr_2>()
// INFRA:   }
// INFRA: }

ssp.library @Lib {
  operator_type @Opr [latency<1>, limit<1>]
}
module @SomeModule {
  ssp.library @Lib {
    operator_type @Opr [latency<2>, limit<2>]
  }
}
ssp.instance @SomeInstance of "ModuloProblem" {
  library @InternalLib {
    operator_type @Opr [latency<3>, limit<3>]
  }
  graph {
    operation<@Opr>()
    operation<@InternalLib::@Opr>()
    operation<@SomeInstance::@InternalLib::@Opr>()
    operation<@Lib::@Opr>()
    operation<@SomeModule::@Lib::@Opr>()
  }
}

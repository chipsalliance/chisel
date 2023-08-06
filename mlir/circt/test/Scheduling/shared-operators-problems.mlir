// RUN: circt-opt %s -ssp-roundtrip=verify
// RUN: circt-opt %s -ssp-schedule=scheduler=simplex | FileCheck %s -check-prefixes=CHECK,SIMPLEX
// RUN: %if or-tools %{ circt-opt %s -ssp-schedule=scheduler=cpsat | FileCheck %s -check-prefixes=CHECK,CPSAT %} 

// CHECK-LABEL: full_load
ssp.instance @full_load of "SharedOperatorsProblem" {
  library {
    operator_type @L1_3 [latency<3>, limit<1>]
    operator_type @_1 [latency<1>]
  }
  graph {
    %0 = operation<@L1_3>() [t<0>]
    %1 = operation<@L1_3>() [t<1>]
    %2 = operation<@L1_3>() [t<2>]
    %3 = operation<@L1_3>() [t<3>]
    %4 = operation<@L1_3>() [t<4>]
    %5 = operation<@_1>(%0, %1, %2, %3, %4) [t<7>]
    // SIMPLEX: @last(%{{.*}}) [t<8>]
    // CPSAT: @last(%{{.*}}) [t<8>]
    operation<@_1> @last(%5) [t<8>]
  }
}

// CHECK-LABEL: partial_load
ssp.instance @partial_load of "SharedOperatorsProblem" {
  library {
    operator_type @L3_3 [latency<3>, limit<3>]
    operator_type @_1 [latency<1>]
  }
  graph {
    %0 = operation<@L3_3>() [t<0>]
    %1 = operation<@L3_3>() [t<1>]
    %2 = operation<@L3_3>() [t<0>]
    %3 = operation<@L3_3>() [t<2>]
    %4 = operation<@L3_3>() [t<1>]
    %5 = operation<@_1>(%0, %1, %2, %3, %4) [t<10>]
    // SIMPLEX: @last(%{{.*}}) [t<5>]
    // CPSAT: @last(%{{.*}}) [t<5>]
    operation<@_1> @last(%5) [t<11>]
  }
}

// CHECK-LABEL: multiple
ssp.instance @multiple of "SharedOperatorsProblem" {
  library {
    operator_type @L3_2 [latency<3>, limit<2>]
    operator_type @L1_1 [latency<1>, limit<1>]
    operator_type @_1 [latency<1>]
  }
  graph {
    %0 = operation<@L3_2>() [t<0>]
    %1 = operation<@L3_2>() [t<1>]
    %2 = operation<@L1_1>() [t<0>]
    %3 = operation<@L3_2>() [t<1>]
    %4 = operation<@L1_1>() [t<1>]
    %5 = operation<@_1>(%0, %1, %2, %3, %4) [t<10>]
    // SIMPLEX: @last(%{{.*}}) [t<5>]
    // CPSAT: @last(%{{.*}}) [t<5>]
    operation<@_1> @last(%5) [t<11>]
  }
}

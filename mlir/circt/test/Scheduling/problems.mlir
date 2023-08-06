// RUN: circt-opt %s -ssp-roundtrip=verify
// RUN: circt-opt %s -ssp-schedule=scheduler=asap | FileCheck %s -check-prefixes=CHECK,ASAP
// RUN: circt-opt %s -ssp-schedule=scheduler=simplex | FileCheck %s -check-prefixes=CHECK,SIMPLEX
// RUN: %if or-tools %{ circt-opt %s -ssp-schedule=scheduler=lp | FileCheck %s -check-prefixes=CHECK,LP %} 

// CHECK-LABEL: unit_latencies
ssp.instance @unit_latencies of "Problem" {
  library {
    operator_type @_1 [latency<1>]
  }
  // ASAP: graph
  graph {
    // ASAP-NEXT: [t<0>]
    %0 = operation<@_1>() [t<0>]
    // ASAP-NEXT: [t<1>]
    %1 = operation<@_1>(%0) [t<1>]
    // ASAP-NEXT: [t<2>]
    %2:3 = operation<@_1>(%0, %1)[t<2>]
    // ASAP-NEXT: [t<3>]
    %3 = operation<@_1>(%2#1) [t<3>]
    // ASAP-NEXT: [t<3>]
    %4 = operation<@_1>(%2#0, %2#2) [t<3>]
    // ASAP-NEXT: [t<4>]
    %5 = operation<@_1>(%3) [t<4>]
    // ASAP-NEXT: [t<5>]
    %6 = operation<@_1>(%3, %4, %5) [t<5>]
    // ASAP-NEXT: [t<6>]
    // SIMPLEX: @last(%{{.*}}) [t<6>]
    // LP: @last(%{{.*}}) [t<6>]
    operation<@_1> @last(%6) [t<6>]
  }
}

// CHECK-LABEL: arbitrary_latencies
ssp.instance @arbitrary_latencies of "Problem" {
  library {
    operator_type @_0 [latency<0>]
    operator_type @_1 [latency<1>]
    operator_type @_3 [latency<3>]
    operator_type @_6 [latency<6>]
    operator_type @_10 [latency<10>]
  }
  // ASAP: graph
  graph {
    // ASAP-NEXT: [t<0>]
    %0 = operation<@_0>() [t<0>]
    // ASAP-NEXT: [t<0>]
    %1 = operation<@_0>() [t<10>]
    // ASAP-NEXT: [t<0>]
    %2 = operation<@_6>(%0) [t<20>]
    // ASAP-NEXT: [t<0>]
    %3 = operation<@_6>(%1) [t<30>]
    // ASAP-NEXT: [t<6>]
    %4 = operation<@_3>(%2, %3) [t<40>]
    // ASAP-NEXT: [t<9>]
    %5 = operation<@_10>(%4) [t<50>]
    // ASAP-NEXT: [t<19>]
    // SIMPLEX: @last(%{{.*}}) [t<19>]
    // LP: @last(%{{.*}}) [t<19>]
    operation<@_1> @last(%5) [t<60>]
  }
}

// CHECK-LABEL: auxiliary_dependences
ssp.instance @auxiliary_dependences of "Problem" {
  library {
    operator_type @_1 [latency<1>]
  }
  // ASAP: graph
  graph {
    // ASAP-NEXT: [t<0>]
    operation<@_1> @op0() [t<0>]
    // ASAP-NEXT: [t<1>]
    operation<@_1> @op1(@op0) [t<1>]
    // ASAP-NEXT: [t<1>]
    operation<@_1> @op2(@op0) [t<1>]
    // ASAP-NEXT: [t<2>]
    operation<@_1> @op3(@op2) [t<2>]
    // ASAP-NEXT: [t<3>]
    operation<@_1> @op4(@op3) [t<3>]
    // ASAP-NEXT: [t<4>]
    operation<@_1> @op5(@op4) [t<4>]
    // ASAP-NEXT: [t<5>]
    // SIMPLEX: @last(@op3, @op5) [t<5>]
    // LP: @last(@op3, @op5) [t<5>]
    operation<@_1> @last(@op3, @op5) [t<5>]
  }
}

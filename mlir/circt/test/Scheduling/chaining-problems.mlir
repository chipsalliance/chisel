// RUN: circt-opt %s -ssp-roundtrip=verify
// RUN: circt-opt %s -ssp-schedule="scheduler=simplex options=cycle-time=5.0" | FileCheck %s -check-prefixes=CHECK,SIMPLEX

// Note: Cycle time is only evaluated for scheduler test; ignored by the problem test!

// CHECK-LABEL: adder_chain
ssp.instance @adder_chain of "ChainingProblem" {
  library {
    operator_type @_0 [latency<0>, incDelay<2.34>, outDelay<2.34>]
    operator_type @_1 [latency<1>, incDelay<0.0>, outDelay<0.0>]
  }
  graph {
    %0 = operation<@_0>() [t<0>, z<0.0>]
    %1 = operation<@_0>(%0) [t<0>, z<2.34>]
    %2 = operation<@_0>(%1) [t<0>, z<4.68>]
    %3 = operation<@_0>(%2) [t<1>, z<0.0>]
    %4 = operation<@_0>(%3) [t<1>, z<2.34>]
    // SIMPLEX: @last(%{{.*}}) [t<2>,
    operation<@_1> @last(%4) [t<2>, z<0.0>]
  }
}

// CHECK-LABEL: multi_cycle
ssp.instance @multi_cycle of "ChainingProblem" {
  library {
    operator_type @_0 [latency<0>, incDelay<2.34>, outDelay<2.34>]
    operator_type @_1 [latency<1>, incDelay<0.0>, outDelay<0.0>]
    operator_type @_3 [latency<3>, incDelay<2.5>, outDelay<3.75>]
  }
  graph {
    %0 = operation<@_0>() [t<0>, z<0.0>]
    %1 = operation<@_0>(%0) [t<0>, z<2.34>]
    %2 = operation<@_3>(%1, %0) [t<0>, z<4.68>]
    %3 = operation<@_0>(%2, %1) [t<3>, z<3.75>]
    %4 = operation<@_0>(%3, %2) [t<3>, z<6.09>]
    // SIMPLEX: @last(%{{.*}}) [t<5>,
    operation<@_1> @last(%4) [t<4>, z<0.0>]
  }
}

// CHECK-LABEL: mco_outgoing_delays
ssp.instance @mco_outgoing_delays of "ChainingProblem" {
  library {
    operator_type @_2 [latency<2>, incDelay<0.1>, outDelay<0.1>]
    operator_type @_3 [latency<3>, incDelay<5.0>, outDelay<0.1>]
  }
  // SIMPLEX: graph
  graph {
    // SIMPLEX-NEXT: [t<0>, z<0.000000e+00 : f32>]
    %0 = operation<@_2>() [t<0>, z<0.0>]

    // Next op cannot start in cycle 2 due to %0's outgoing delay: 0.1+5.0 > 5.0.
    // SIMPLEX-NEXT: [t<3>, z<0.000000e+00 : f32>]
    %1 = operation<@_3>(%0) [t<3>, z<0.0>]
    
    // SIMPLEX-NEXT: [t<6>, z<1.000000e-01 : f32>]
    %2 = operation<@_2>(%1) [t<6>, z<0.1>]

    // Next op should have SITC=0.1 (not: 0.2), because we only consider %2's outgoing delay.
    // SIMPLEX-NEXT: [t<8>, z<1.000000e-01 : f32>]
    operation<@_2> @last(%2) [t<8>, z<0.1>]
  }
}

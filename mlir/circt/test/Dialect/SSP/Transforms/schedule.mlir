// RUN: circt-opt %s -ssp-schedule="scheduler=simplex options=last-op-name=sink,cycle-time=5.0" | circt-opt | FileCheck %s

// CHECK: ssp.instance @self_arc of "CyclicProblem" [II<3>] {
// CHECK:   library {
// CHECK:     operator_type @unit [latency<1>]
// CHECK:     operator_type @_3 [latency<3>]
// CHECK:   }
// CHECK:   graph {
// CHECK:     %[[op_0:.*]] = operation<@unit>() [t<0>]
// CHECK:     %[[op_1:.*]] = operation<@_3> @self(%[[op_0]], %[[op_0]], @self [dist<1>]) [t<1>]
// CHECK:     operation<@unit> @sink(%[[op_1]]) [t<4>]
// CHECK:   }
// CHECK: }
ssp.instance @self_arc of "CyclicProblem" {
  library {
    operator_type @unit [latency<1>]
    operator_type @_3 [latency<3>]
  }
  graph {
    %0 = operation<@unit>()
    %1 = operation<@_3> @self(%0, %0, @self [dist<1>])
    operation<@unit> @sink(%1)
  }
}

// CHECK: ssp.instance @mco_outgoing_delays of "ChainingProblem" {
// CHECK:   library {
// CHECK:     operator_type @add [latency<2>, incDelay<1.000000e-01 : f32>, outDelay<1.000000e-01 : f32>]
// CHECK:     operator_type @mul [latency<3>, incDelay<5.000000e+00 : f32>, outDelay<1.000000e-01 : f32>]
// CHECK:     operator_type @ret [latency<0>, incDelay<0.000000e+00 : f32>, outDelay<0.000000e+00 : f32>]
// CHECK:   }
// CHECK:   graph {
// CHECK:     %[[op_0:.*]] = operation<@add>() [t<0>, z<0.000000e+00 : f32>]
// CHECK:     %[[op_1:.*]] = operation<@mul>(%[[op_0]], %[[op_0]]) [t<3>, z<0.000000e+00 : f32>]
// CHECK:     %[[op_2:.*]] = operation<@add>(%[[op_1]], %[[op_1]]) [t<6>, z<1.000000e-01 : f32>]
// CHECK:     operation<@ret> @sink(%[[op_2]]) [t<8>, z<1.000000e-01 : f32>]
// CHECK:   }
// CHECK: }
ssp.instance @mco_outgoing_delays of "ChainingProblem" {
  library {
    operator_type @add [latency<2>, incDelay<1.000000e-01 : f32>, outDelay<1.000000e-01 : f32>]
    operator_type @mul [latency<3>, incDelay<5.000000e+00 : f32>, outDelay<1.000000e-01 : f32>]
    operator_type @ret [latency<0>, incDelay<0.000000e+00 : f32>, outDelay<0.000000e+00 : f32>]
  }
  graph {
    %0 = operation<@add>()
    %1 = operation<@mul>(%0, %0)
    %2 = operation<@add>(%1, %1)
    operation<@ret> @sink(%2)
  }
}

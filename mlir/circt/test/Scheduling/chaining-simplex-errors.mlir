// RUN: circt-opt %s -ssp-schedule="scheduler=simplex options=cycle-time=2.0" -verify-diagnostics -split-input-file

// expected-error@+1 {{Delays of operator type 'inv' exceed maximum cycle time}}
ssp.instance @invalid_delay of "ChainingProblem" {
  library {
    operator_type @inv [latency<0>, incDelay<2.34>, outDelay<2.34>]
  }
  graph {
    operation<@inv>()
  }
}

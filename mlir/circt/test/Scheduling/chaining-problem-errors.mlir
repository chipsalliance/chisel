// RUN: circt-opt %s -ssp-roundtrip=verify -verify-diagnostics -split-input-file

// expected-error@+1 {{Missing delays}}
ssp.instance @missing_delay of "ChainingProblem" {
  library {
    operator_type @_0 [latency<0>]
  }
  graph {}
}

// -----

// expected-error@+1 {{Negative delays}}
ssp.instance @negative_delay of "ChainingProblem" {
  library {
    operator_type @_0 [latency<0>, incDelay<-1.0>, outDelay<-1.0>]
  }
  graph {}
}

// -----

// expected-error@+1 {{Incoming & outgoing delay must be equal for zero-latency operator type}}
ssp.instance @inc_out_mismatch of "ChainingProblem" {
  library {
    operator_type @_0 [latency<0>, incDelay<1.0>, outDelay<2.0>]
  }
  graph {}
}

// -----

ssp.instance @no_stic of "ChainingProblem" {
  library {
    operator_type @_0 [latency<0>, incDelay<1.0>, outDelay<1.0>]
  }
  graph {
    operation<@_0>() [t<0>] // expected-error {{Operation has no non-negative start time in cycle}}
  }
}

// -----

// expected-error@+1 {{Precedence violated in cycle 0}}
ssp.instance @precedence1 of "ChainingProblem" {
  library {
    operator_type @_0 [latency<0>, incDelay<1.0>, outDelay<1.0>]
  }
  graph {
    %0 = operation<@_0>() [t<0>, z<1.1>]
    operation<@_0>(%0) [t<0>, z<2.0>]
  }
}

// -----

// expected-error@+1 {{Precedence violated in cycle 3}}
ssp.instance @precedence2 of "ChainingProblem" {
  library {
    operator_type @_0 [latency<0>, incDelay<1.0>, outDelay<1.0>]
    operator_type @_3 [latency<3>, incDelay<2.5>, outDelay<3.75>]
  }
  graph {
    %0 = operation<@_3>() [t<0>, z<0.0>]
    operation<@_0>(%0) [t<3>, z<3.0>]
  }
}

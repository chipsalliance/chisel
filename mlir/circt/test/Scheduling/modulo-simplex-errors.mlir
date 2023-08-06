// RUN: circt-opt %s -ssp-schedule="scheduler=simplex options=last-op-name=last" -verify-diagnostics -split-input-file

// expected-error@+1 {{last operation is not a sink}}
ssp.instance of "ModuloProblem" {
  library {
    operator_type @_1 [latency<1>]
  }
  graph {
    operation<@_1> @last()
    operation<@_1>(@last)
  }
}

// -----

// expected-error@+1 {{multiple sinks detected}}
ssp.instance of "ModuloProblem" {
  library {
    operator_type @_1 [latency<1>]
  }
  graph {
    operation<@_1>()
    operation<@_1> @last()
  }
}

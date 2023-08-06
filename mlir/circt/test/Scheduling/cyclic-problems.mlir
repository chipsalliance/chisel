// RUN: circt-opt %s -ssp-roundtrip=verify
// RUN: circt-opt %s -ssp-schedule=scheduler=simplex | FileCheck %s -check-prefixes=CHECK,SIMPLEX
// RUN: %if or-tools %{ circt-opt %s -ssp-schedule=scheduler=lp | FileCheck %s -check-prefixes=CHECK,LP %} 

// CHECK-LABEL: cyclic
// SIMPLEX-SAME: [II<2>]
// LP-SAME: [II<2>]
ssp.instance @cyclic of "CyclicProblem" [II<2>] {
  library {
    operator_type @_0 [latency<0>]
    operator_type @_1 [latency<1>]
    operator_type @_2 [latency<2>]
  }
  graph {
    %0 = operation<@_1>() [t<0>]
    %1 = operation<@_0>(@op4 [dist<1>]) [t<2>]
    %2 = operation<@_2>(@op4 [dist<2>]) [t<0>]
    %3 = operation<@_1>(%1, %2) [t<2>]
    %4 = operation<@_1> @op4(%2, %0) [t<3>]
    // SIMPLEX: @last(%{{.*}}) [t<3>]
    // LP: @last(%{{.*}}) [t<3>]
    operation<@_1> @last(%4) [t<4>]
  }
}

// CHECK-LABEL: mobility
// SIMPLEX-SAME: [II<3>]
// LP-SAME: [II<3>]
ssp.instance @mobility of "CyclicProblem" [II<3>] {
  library {
    operator_type @_1 [latency<1>]
    operator_type @_4 [latency<4>]
  }
  graph {
    %0 = operation<@_1>() [t<0>]
    %1 = operation<@_4>(%0) [t<1>]
    %2 = operation<@_1>(%0, @op5 [dist<1>]) [t<4>]
    %3 = operation<@_1>(%1, %2) [t<5>]
    %4 = operation<@_4>(%3) [t<6>]
    %5 = operation<@_1> @op5(%3) [t<6>]
    // SIMPLEX: @last(%{{.*}}, %{{.*}}) [t<10>]
    // LP: @last(%{{.*}}, %{{.*}}) [t<10>]
    operation<@_1> @last(%4, %5) [t<10>]
  }
}

// CHECK-LABEL: interleaved_cycles
// SIMPLEX-SAME: [II<4>]
// LP-SAME: [II<4>]
ssp.instance @interleaved_cycles of "CyclicProblem" [II<4>] {
  library {
    operator_type @_1 [latency<1>]
    operator_type @_10 [latency<10>]
  }
  graph {
    %0 = operation<@_1>() [t<0>]
    %1 = operation<@_10>(%0) [t<1>]
    %2 = operation<@_1>(%0, @op6 [dist<2>]) [t<10>]
    %3 = operation<@_1>(%1, %2) [t<11>]
    %4 = operation<@_10>(%3) [t<12>]
    %5 = operation<@_1>(%3, @op9 [dist<2>]) [t<16>]
    %6 = operation<@_1> @op6(%5) [t<17>]
    %7 = operation<@_1>(%4, %6) [t<22>]
    %8 = operation<@_10>(%7) [t<23>]
    %9 = operation<@_1> @op9(%7) [t<23>]
    // SIMPLEX: @last(%{{.*}}, %{{.*}}) [t<33>]
    // LP: @last(%{{.*}}, %{{.*}}) [t<33>]
    operation<@_1> @last(%8, %9) [t<33>]
  }
}

// CHECK-LABEL: self_arc
// SIMPLEX-SAME: [II<3>]
// LP-SAME: [II<3>]
ssp.instance @self_arc of "CyclicProblem" [II<3>] {
  library {
    operator_type @_1 [latency<1>]
    operator_type @_3 [latency<3>]
  }
  graph {
    %0 = operation<@_1>() [t<0>]
    %1 = operation<@_3> @op1(%0, @op1 [dist<1>]) [t<1>]
    // SIMPLEX: operation<@_1> @last(%{{.*}}) [t<4>]
    // LP: operation<@_1> @last(%{{.*}}) [t<4>]
    %2 = operation<@_1> @last(%1) [t<4>]
  }
}

// RUN: circt-opt -verify-diagnostics -pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-lower-chirrtl)))'  %s

firrtl.circuit "NoInferredEnables" {
firrtl.module @NoInferredEnables(in %p: !firrtl.uint<1>, in %addr: !firrtl.uint<4>, in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %v: !firrtl.uint<32>) {
  %ram = chirrtl.seqmem Undefined  : !chirrtl.cmemory<uint<32>, 16>
  %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
  %r = firrtl.regreset %clock, %reset, %c0_ui4  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>
  // expected-warning @+1 {{memory port is never enabled}}
  %ramport_data, %ramport_port = chirrtl.memoryport Read %ram {name = "ramport"} : (!chirrtl.cmemory<uint<32>, 16>) -> (!firrtl.uint<32>, !chirrtl.cmemoryport)
  chirrtl.memoryport.access %ramport_port[%addr], %clock : !chirrtl.cmemoryport, !firrtl.uint<4>, !firrtl.clock

  firrtl.connect %v, %ramport_data : !firrtl.uint<32>, !firrtl.uint<32>
}
}

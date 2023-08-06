// RUN: circt-opt %s --convert-to-arcs --verify-diagnostics

hw.module @Empty() {
  // expected-error @+1 {{op has regions; not supported by ConvertToArcs}}
  scf.execute_region {
    scf.yield
  }
}

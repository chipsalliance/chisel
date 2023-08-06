// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-types{preserve-aggregate=all}))' %s    | FileCheck --check-prefix=PRESERVE_ALL %s
// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-types{preserve-aggregate=vec}))' %s    | FileCheck --check-prefix=PRESERVE_VEC %s
// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-types{preserve-aggregate=1d-vec}))' %s | FileCheck --check-prefix=PRESERVE_1D_VEC %s
// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-types{preserve-aggregate=none}))' %s   | FileCheck --check-prefix=PRESERVE_NONE %s

firrtl.circuit "TopLevel" {
  firrtl.module @TopLevel() {}

  // * A module using the internal convention will have its ports lowered
  //   according to the preservation mode.

  // * A module using the scalarized convention will always have it's ports
  //   fully scalarized, regardless of the preservation mode.

  // 1D Vector Ports

  // PRESERVE_ALL:    @InternalModule1(in %port: !firrtl.vector<uint<8>, 2>)
  // PRESERVE_VEC:    @InternalModule1(in %port: !firrtl.vector<uint<8>, 2>)
  // PRESERVE_1D_VEC: @InternalModule1(in %port: !firrtl.vector<uint<8>, 2>)
  // PRESERVE_NONE:   @InternalModule1(in %port_0: !firrtl.uint<8>, in %port_1: !firrtl.uint<8>)
  firrtl.module @InternalModule1(in %port: !firrtl.vector<uint<8>, 2>)
    attributes {convention = #firrtl<convention internal>} {}

  // PRESERVE_ALL:    @ScalarizedModule1(in %port_0: !firrtl.uint<8>, in %port_1: !firrtl.uint<8>)
  // PRESERVE_VEC:    @ScalarizedModule1(in %port_0: !firrtl.uint<8>, in %port_1: !firrtl.uint<8>)
  // PRESERVE_1D_VEC: @ScalarizedModule1(in %port_0: !firrtl.uint<8>, in %port_1: !firrtl.uint<8>)
  // PRESERVE_NONE:   @ScalarizedModule1(in %port_0: !firrtl.uint<8>, in %port_1: !firrtl.uint<8>)
  firrtl.module @ScalarizedModule1(in %port: !firrtl.vector<uint<8>, 2>)
    attributes {convention = #firrtl<convention scalarized>} {}
  
  // 2D Vector Ports

  // PRESERVE_ALL:    @InternalModule2(in %port: !firrtl.vector<vector<uint<8>, 2>, 2>)
  // PRESERVE_VEC:    @InternalModule2(in %port: !firrtl.vector<vector<uint<8>, 2>, 2>)
  // PRESERVE_1D_VEC: @InternalModule2(in %port_0: !firrtl.vector<uint<8>, 2>, in %port_1: !firrtl.vector<uint<8>, 2>)
  // PRESERVE_NONE:   @InternalModule2(in %port_0_0: !firrtl.uint<8>, in %port_0_1: !firrtl.uint<8>, in %port_1_0: !firrtl.uint<8>, in %port_1_1: !firrtl.uint<8>)
  firrtl.module @InternalModule2(in %port: !firrtl.vector<vector<uint<8>, 2>, 2>)
    attributes {convention = #firrtl<convention internal>} {}
  
  // PRESERVE_ALL:    ScalarizedModule2(in %port_0_0: !firrtl.uint<8>, in %port_0_1: !firrtl.uint<8>, in %port_1_0: !firrtl.uint<8>, in %port_1_1: !firrtl.uint<8>)
  // PRESERVE_VEC:    ScalarizedModule2(in %port_0_0: !firrtl.uint<8>, in %port_0_1: !firrtl.uint<8>, in %port_1_0: !firrtl.uint<8>, in %port_1_1: !firrtl.uint<8>)
  // PRESERVE_1D_VEC: ScalarizedModule2(in %port_0_0: !firrtl.uint<8>, in %port_0_1: !firrtl.uint<8>, in %port_1_0: !firrtl.uint<8>, in %port_1_1: !firrtl.uint<8>)
  // PRESERVE_NONE:   ScalarizedModule2(in %port_0_0: !firrtl.uint<8>, in %port_0_1: !firrtl.uint<8>, in %port_1_0: !firrtl.uint<8>, in %port_1_1: !firrtl.uint<8>)
  firrtl.module @ScalarizedModule2(in %port: !firrtl.vector<vector<uint<8>, 2>, 2>)
    attributes {convention = #firrtl<convention scalarized>} {}

  // Bundle Ports
      
  // PRESERVE_ALL:    @InternalModule3(in %port: !firrtl.bundle<field: uint<1>>)
  // PRESERVE_VEC:    @InternalModule3(in %port_field: !firrtl.uint<1>)
  // PRESERVE_1D_VEC: @InternalModule3(in %port_field: !firrtl.uint<1>)
  // PRESERVE_NONE:   @InternalModule3(in %port_field: !firrtl.uint<1>)
  firrtl.module @InternalModule3(in %port: !firrtl.bundle<field: uint<1>>)
    attributes {convention = #firrtl<convention internal>} {}
  
  // PRESERVE_ALL:    @ScalarizedModule3(in %port_field: !firrtl.uint<1>)
  // PRESERVE_VEC:    @ScalarizedModule3(in %port_field: !firrtl.uint<1>)
  // PRESERVE_1D_VEC: @ScalarizedModule3(in %port_field: !firrtl.uint<1>)
  // PRESERVE_NONE:   @ScalarizedModule3(in %port_field: !firrtl.uint<1>)
  firrtl.module @ScalarizedModule3(in %port: !firrtl.bundle<field: uint<1>>)
    attributes {convention = #firrtl<convention scalarized>} {}
}

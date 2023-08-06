// RUN: circt-opt -lower-firrtl-to-hw %s | FileCheck %s

firrtl.circuit "Arithmetic" {
  // CHECK-LABEL: hw.module @Arithmetic
  firrtl.module @Arithmetic(in %uin3c: !firrtl.uint<3>,
                            out %out0: !firrtl.uint<3>,
                            out %out1: !firrtl.uint<4>,
                            out %out2: !firrtl.uint<4>,
                            out %out3: !firrtl.uint<1>) {
  %uin0c = firrtl.wire : !firrtl.uint<0>

    // CHECK-DAG: [[MULZERO:%.+]] = hw.constant 0 : i3
    %0 = firrtl.mul %uin0c, %uin3c : (!firrtl.uint<0>, !firrtl.uint<3>) -> !firrtl.uint<3>
    firrtl.connect %out0, %0 : !firrtl.uint<3>, !firrtl.uint<3>

    // Lowers to nothing.
    %m0 = firrtl.mul %uin0c, %uin0c : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>

    // Lowers to nothing.
    %node = firrtl.node %m0 : !firrtl.uint<0>

    // Lowers to nothing.  Issue #429.
    %div = firrtl.div %node, %uin3c : (!firrtl.uint<0>, !firrtl.uint<3>) -> !firrtl.uint<0>

    // CHECK-DAG: %c0_i4 = hw.constant 0 : i4
    // CHECK-DAG: %false = hw.constant false
    // CHECK-NEXT: [[UIN3EXT:%.+]] = comb.concat %false, %uin3c : i1, i3
    // CHECK-NEXT: [[ADDRES:%.+]] = comb.add bin [[UIN3EXT]], %c0_i4 : i4
    %1 = firrtl.add %uin0c, %uin3c : (!firrtl.uint<0>, !firrtl.uint<3>) -> !firrtl.uint<4>
    firrtl.connect %out1, %1 : !firrtl.uint<4>, !firrtl.uint<4>

    %2 = firrtl.shl %node, 4 : (!firrtl.uint<0>) -> !firrtl.uint<4>
    firrtl.connect %out2, %2 : !firrtl.uint<4>, !firrtl.uint<4>

    // Issue #436
    %3 = firrtl.eq %uin0c, %uin0c : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
    firrtl.connect %out3, %3 : !firrtl.uint<1>, !firrtl.uint<1>

    // CHECK: hw.output %c0_i3, [[ADDRES]], %c0_i4, %true
  }

  // CHECK-LABEL: hw.module private @Exotic
  firrtl.module private @Exotic(in %uin3c: !firrtl.uint<3>,
                        out %out0: !firrtl.uint<3>,
                        out %out1: !firrtl.uint<3>) {
    %uin0c = firrtl.wire : !firrtl.uint<0>

    // CHECK-DAG: = hw.constant true
    %0 = firrtl.andr %uin0c : (!firrtl.uint<0>) -> !firrtl.uint<1>

    // CHECK-DAG: = hw.constant false
    %1 = firrtl.xorr %uin0c : (!firrtl.uint<0>) -> !firrtl.uint<1>

    %2 = firrtl.orr %uin0c : (!firrtl.uint<0>) -> !firrtl.uint<1>

    // Lowers to the uin3 value.
    %3 = firrtl.cat %uin0c, %uin3c : (!firrtl.uint<0>, !firrtl.uint<3>) -> !firrtl.uint<3>
    firrtl.connect %out0, %3 : !firrtl.uint<3>, !firrtl.uint<3>

    // Lowers to the uin3 value.
    %4 = firrtl.cat %uin3c, %uin0c : (!firrtl.uint<3>, !firrtl.uint<0>) -> !firrtl.uint<3>
    firrtl.connect %out1, %4 : !firrtl.uint<3>, !firrtl.uint<3>

    // Lowers to nothing.
    %5 = firrtl.cat %uin0c, %uin0c : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>

    // CHECK: hw.output %uin3c, %uin3c : i3, i3
  }

  // CHECK-LABEL: hw.module private @Decls
  firrtl.module private @Decls(in %uin3c: !firrtl.uint<3>) {
    %sin0c = firrtl.wire : !firrtl.sint<0>
    %uin0c = firrtl.wire : !firrtl.uint<0>

    // Lowers to nothing.
    %wire = firrtl.wire : !firrtl.sint<0>
    firrtl.connect %wire, %sin0c : !firrtl.sint<0>, !firrtl.sint<0>

    // CHECK-NEXT: hw.output
  }

}

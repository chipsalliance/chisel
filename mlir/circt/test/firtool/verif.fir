; RUN: firtool %s | FileCheck %s

circuit Foo:
  intmodule LTLDelay:
    input in: UInt<1>
    output out: UInt<1>
    intrinsic = circt_ltl_delay
    parameter delay = 1
    parameter length = 0

  intmodule LTLConcat:
    input lhs: UInt<1>
    input rhs: UInt<1>
    output out: UInt<1>
    intrinsic = circt_ltl_concat

  intmodule LTLImplication:
    input lhs: UInt<1>
    input rhs: UInt<1>
    output out: UInt<1>
    intrinsic = circt_ltl_implication

  intmodule LTLEventually:
    input in: UInt<1>
    output out: UInt<1>
    intrinsic = circt_ltl_eventually

  intmodule VerifAssert:
    input property: UInt<1>
    intrinsic = circt_verif_assert
    parameter label = "hello"

  module Foo:
    input clk: Clock
    input a: UInt<1>
    input b: UInt<1>

    inst delay of LTLDelay
    delay.in <= b

    inst concat of LTLConcat
    concat.lhs <= a
    concat.rhs <= delay.out

    inst implication of LTLImplication
    implication.lhs <= a
    implication.rhs <= concat.out

    inst eventually of LTLEventually
    eventually.in <= implication.out

    ; CHECK: hello: assert property (s_eventually a |-> a ##1 b);
    inst assert of VerifAssert
    assert.property <= eventually.out


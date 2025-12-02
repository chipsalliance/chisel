// RUN: scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" %s | FileCheck %s
// SPDX-License-Identifier: Apache-2.0

import chisel3._
import chisel3.probe._

// CHECK-LABEL: public module HelloWorld :
class HelloWorld extends Module {
  // CHECK: printf(clock, UInt<1>(0h1), "Hello World!\n")
  printf("Hello World!\n")
  // CHECK: stop(clock, UInt<1>(0h1), 0)
  stop()
}

println(circt.stage.ChiselStage.emitCHIRRTL(new HelloWorld))

// CHECK-LABEL: public module Verf :
class Verf extends Module {
  val i = IO(Input(UInt(8.W)))
  val o = IO(Output(UInt(8.W)))
  // CHECK: connect o, i
  o := i
  // CHECK: cover(clock, _cov_T,
  val cov = chisel3.cover(i === 1.U)
  // CHECK: assume(clock, _assm_T,
  val assm = chisel3.assume(i =/= 8.U)
  // CHECK: intrinsic(circt_chisel_ifelsefatal<
  // CHECK: clock, _asst_T, _asst_T_1
  val asst = chisel3.assert(o === i)
}

println(circt.stage.ChiselStage.emitCHIRRTL(new Verf))

// Following test ported from ProbeSpec.scala in chisel test suite
// CHECK-LABEL: circuit Probe :
class Probe extends Module {
  val x = IO(Input(Bool()))
  val y = IO(Output(Bool()))

  // CHECK: module UTurn :
  class UTurn extends RawModule {
    // CHECK-NEXT: input in : RWProbe<UInt<1>>
    val in = IO(Input(RWProbe(Bool())))
    // CHECK-NEXT: output out : RWProbe<UInt<1>>
    val out = IO(Output(RWProbe(Bool())))
    probe.define(out, in)
  }

  // CHECK-LABEL: inst u1 of UTurn
  val u1 = Module(new UTurn)
  val u2 = Module(new UTurn)

  val n = RWProbeValue(x)

  // CHECK: define u1.in = rwprobe(x)
  probe.define(u1.in, n)
  // CHECK: define u2.in = u1.out
  probe.define(u2.in, u1.out)

  // CHECK: connect y, read(u2.out)
  y := read(u2.out)
  // CHECK: force_initial(u1.out, UInt<1>(0h0))
  probe.forceInitial(u1.out, false.B)
  // CHECK: release_initial(u1.out)
  probe.releaseInitial(u1.out)

  when(x) {
    // CHECK: force(clock, _T, u2.out, UInt<1>(0h0))
    probe.force(u2.out, false.B)
  }
  when(y) {
    // CHECK: release(clock, _T_1, u2.out)
    probe.release(u2.out)
  }
}

println(circt.stage.ChiselStage.emitCHIRRTL(new Probe))

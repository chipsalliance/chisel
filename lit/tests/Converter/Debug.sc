// RUN: scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" --java-opt="--enable-native-access=ALL-UNNAMED" --java-opt="--enable-preview" --java-opt="-Djava.library.path=%JAVALIBRARYPATH" %s | FileCheck %s -check-prefix=FIRRTL
// SPDX-License-Identifier: Apache-2.0

import chisel3._
import chisel3.probe._

// FIRRTL-LABEL: public module HelloWorld :
class HelloWorld extends Module {
  // FIRRTL: printf(clock, UInt<1>(1), "Hello World!\n")
  printf("Hello World!\n")
  // FIRRTL: stop(clock, UInt<1>(1), 0)
  stop()
}

println(lit.utility.panamaconverter.firrtlString(new HelloWorld))

// FIRRTL-LABEL: public module Verf :
class Verf extends Module {
  val i = IO(Input(UInt(8.W)))
  val o = IO(Output(UInt(8.W)))
  // FIRRTL: connect o, i
  o := i
  // FIRRTL: cover(clock, _cov_T,
  val cov = chisel3.cover(i === 1.U)
  // FIRRTL: assume(clock, _assm_T,
  val assm = chisel3.assume(i =/= 8.U)
  // FIRRTL: intrinsic(circt_chisel_ifelsefatal<
  // FIRRTL: clock, _asst_T, _asst_T_1
  val asst = chisel3.assert(o === i)
}

// NOTE: currently CIRCT emits cover, assume but PanamaConverter does not
println(lit.utility.panamaconverter.firrtlString(new Verf))

// Following test ported from ProbeSpec.scala in chisel test suite
// FIRRTL-LABEL: circuit Probe :
class Probe extends Module {
  val x = IO(Input(Bool()))
  val y = IO(Output(Bool()))

  // FIRRTL: module UTurn :
  class UTurn extends RawModule {
    // FIRRTL-NEXT: input in : RWProbe<UInt<1>>
    val in = IO(Input(RWProbe(Bool())))
    // FIRRTL-NEXT: output out : RWProbe<UInt<1>>
    val out = IO(Output(RWProbe(Bool())))
    probe.define(out, in)
  }

  // FIRRTL-LABEL: inst u1 of UTurn
  val u1 = Module(new UTurn)
  val u2 = Module(new UTurn)

  val n = RWProbeValue(x)

  // FIRRTL: define u1.in = rwprobe(x)
  probe.define(u1.in, n)
  // FIRRTL: define u2.in = u1.out
  probe.define(u2.in, u1.out)

  // FIRRTL: connect y, read(u2.out)
  y := read(u2.out)
  // FIRRTL: force_initial(u1.out, UInt<1>(0))
  probe.forceInitial(u1.out, false.B)
  // FIRRTL: release_initial(u1.out)
  probe.releaseInitial(u1.out)

  when(x) {
    // FIRRTL: force(clock, _T, u2.out, UInt<1>(0))
    probe.force(u2.out, false.B)
  }
  when(y) {
    // FIRRTL: release(clock, _T_1, u2.out)
    probe.release(u2.out)
  }
}

println(lit.utility.panamaconverter.firrtlString(new Probe))

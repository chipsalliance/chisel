// RUN: scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" %s | FileCheck %s
// SPDX-License-Identifier: Apache-2.0

import chisel3._
import chisel3.util.circt.IsX


// CHECK-LABEL: circuit FooModule :
// CHECK: extmodule FooBlackbox :
// CHECK-NEXT: output o : UInt<1>
// CHECK-NEXT: defname = FooBlackbox
class FooBlackbox extends BlackBox {
  val io = IO(new Bundle{
    val o = Output(Bool())
  })
}

// CHECK: public module FooModule :
// CHECK-NEXT: input clock : Clock
// CHECK-NEXT: input reset : UInt<1>
// CHECK-NEXT: output o : UInt<1>
class FooModule extends Module {
  val o = IO(Output(Bool()))

  // CHECK: inst bb of FooBlackbox
  val bb = Module(new FooBlackbox)

  // CHECK-NEXT: node _o_T = intrinsic(circt_isX : UInt<1>, bb.o)
  // CHECK-NEXT: connect o, _o_T
  o := IsX(bb.io.o)
}

println(circt.stage.ChiselStage.emitCHIRRTL(new FooModule))

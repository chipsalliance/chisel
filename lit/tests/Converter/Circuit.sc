// RUN: scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" --java-opt="--enable-native-access=ALL-UNNAMED" --java-opt="--enable-preview" --java-opt="-Djava.library.path=%JAVALIBRARYPATH" %s | FileCheck %s -check-prefix=FIRRTL
// SPDX-License-Identifier: Apache-2.0

import chisel3._
import chisel3.util.circt.IsX

// FIRRTL-LABEL: circuit FooModule :
// FIRRTL-NEXT: extmodule FooBlackbox :
// FIRRTL-NEXT: output o : UInt<1>
// FIRRTL-NEXT: defname = FooBlackbox
class FooBlackbox extends BlackBox {
  val io = IO(new Bundle {
    val o = Output(Bool())
  })
}

// FIRRTL: public module FooModule :
// FIRRTL-NEXT: input clock : Clock
// FIRRTL-NEXT: input reset : UInt<1>
// FIRRTL-NEXT: output o : UInt<1>
class FooModule extends Module {
  val o = IO(Output(Bool()))

  // FIRRTL: inst bb of FooBlackbox
  val bb = Module(new FooBlackbox)

  // FIRRTL-NEXT: connect o, intrinsic(circt_isX : UInt<1>, bb.o)
  o := IsX(bb.io.o)
}

println(lit.utility.panamaconverter.firrtlString(new FooModule))

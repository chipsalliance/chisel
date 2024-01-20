// RUN: scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" %s | FileCheck %s

import chisel3._
import circt.stage.ChiselStage
class FooBundle extends Bundle {
  val foo = Input(UInt(3.W))
}
// CHECK-LABEL: module FooModule
// CHECK: input clock : Clock
// CHECK: input reset : UInt<1>
class FooModule extends Module {
  // CHECK: output io : { flip foo : UInt<3>} 
  val io = IO(new FooBundle)
  // CHECK: skip
}
println(ChiselStage.emitCHIRRTL(new FooModule))

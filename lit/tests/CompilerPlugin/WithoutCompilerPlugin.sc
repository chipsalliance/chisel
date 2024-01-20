// RUN: not scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --java-opt="--enable-native-access=ALL-UNNAMED --enable-preview -Djava.library.path=%JAVALIBRARYPATH" %s 2>&1 | FileCheck %s

import chisel3._
import circt.stage.ChiselStage
class FooBundle extends Bundle {
  val foo = Input(UInt(3.W))
}
class FooModule extends Module {
  // CHECK: assertion failed: The Chisel compiler plugin is now required for compiling Chisel code.
  val io = IO(new FooBundle)
}
println(ChiselStage.emitCHIRRTL(new FooModule))

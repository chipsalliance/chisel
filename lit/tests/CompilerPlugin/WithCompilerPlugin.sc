// RUN: scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" --java-opt="--enable-native-access=ALL-UNNAMED --enable-preview -Djava.library.path=%JAVALIBRARYPATH" %s

import chisel3._
import circt.stage.ChiselStage
class FooBundle extends Bundle {
  val foo = Input(UInt(3.W))
}
class FooModule extends Module {
  val io = IO(new FooBundle)
}
println(ChiselStage.emitCHIRRTL(new FooModule))

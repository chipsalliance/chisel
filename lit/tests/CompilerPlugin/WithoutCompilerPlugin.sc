//> using javaHome $JAVAHOME
//> using jars $RUNCLASSPATH
//> using scala $SCALAVERSION
//> using options $PLUGINJARS
//> using javaOpt --enable-native-access=ALL-UNNAMED --enable-preview $JAVALIBRARYPATH

import chisel3._
import circt.stage.ChiselStage
class FooBundle extends Bundle {
  val foo = Input(UInt(3.W))
}
class FooModule extends Module {
  val io = IO(new FooBundle)
}
println(ChiselStage.emitCHIRRTL(new FooModule))

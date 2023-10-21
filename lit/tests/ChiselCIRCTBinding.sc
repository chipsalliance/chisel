//> using javaHome $JAVAHOME
//> using jars $RUNCLASSPATH
//> using scala $SCALAVERSION
//> using options $PLUGINJARS
//> using javaOpt --enable-native-access=ALL-UNNAMED --enable-preview $JAVALIBRARYPATH

import chisel3._
import utility.binding._

class FooBundle extends Bundle {
  val foo = Input(UInt(3.W))
}
class FooModule extends Module {
  val io = IO(new FooBundle)
}
firrtlString(new FooModule)
verilogString(new FooModule)
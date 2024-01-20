// RUN: scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" --java-opt="--enable-native-access=ALL-UNNAMED" --java-opt="--enable-preview" --java-opt="-Djava.library.path=%JAVALIBRARYPATH" %s | FileCheck %s

import chisel3._
import lit.utility.panamaconverter._

class FooBundle extends Bundle {
  val foo = Input(UInt(3.W))
}
// CHECK: circuit FooModule :
// CHECK: module FooModule :
// CHECK: input clock : Clock
// CHECK: input reset : UInt<1>
class FooModule extends Module {
  // CHECK: output io : { flip foo : UInt<3> }
  val io = IO(new FooBundle)
}
print(firrtlString(new FooModule))
// RUN: scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" --java-opt="--enable-native-access=ALL-UNNAMED" --java-opt="--enable-preview" --java-opt="-Djava.library.path=%JAVALIBRARYPATH" %s
// SPDX-License-Identifier: Apache-2.0

import chisel3._
import chisel3.experimental.{Analog, attach}
import chisel3.util.SRAM

class Mem extends Module {
  val sr = SRAM(1024, UInt(8.W), 1, 1, 1)
}

println(circt.stage.ChiselStage.emitCHIRRTL(new Mem))
println(lit.utility.panamaconverter.firrtlString(new Mem))

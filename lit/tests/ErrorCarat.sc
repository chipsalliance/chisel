// RUN: env JDK_JAVA_OPTIONS='-Dchisel.project.root=' not scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" %s | FileCheck %s
// RUN: env JDK_JAVA_OPTIONS='-Dchisel.project.root=%S' not scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" --scala-option="-Xmacro-settings:chisel.project.root=" %s | FileCheck %s
// SPDX-License-Identifier: Apache-2.0

import chisel3._

class ModuleWithError extends Module {
  val in = IO(Input(UInt(8.W)))
  val out = IO(Output(UInt(16.W)))

  // Check absolute path source locators with an empty project root via either setting.
  // CHECK:      High index 15 is out of range [0, 7]
  // CHECK-NEXT: out := in(15, 0)
  // CHECK-NEXT:          ^
  out := in(15, 0)
}

circt.stage.ChiselStage.emitCHIRRTL(new ModuleWithError)

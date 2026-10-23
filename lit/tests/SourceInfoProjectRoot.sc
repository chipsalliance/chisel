// RUN: env JDK_JAVA_OPTIONS='-Dchisel.project.root=' scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" --scala-option="-Xmacro-settings:unrelated=value,chisel.project.root=%S" %s | FileCheck %s
// RUN: env JDK_JAVA_OPTIONS='-Dchisel.project.root=' scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" --scala-option="-Xmacro-settings:chisel.project.root=%S/" %s | FileCheck %s
// RUN: env JDK_JAVA_OPTIONS='-Dchisel.project.root=%S' scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" --scala-option="-Xmacro-settings:unrelated=value" %s | FileCheck %s
// SPDX-License-Identifier: Apache-2.0

import chisel3._

// The module locator comes from the compiler plugin.
// Scala CLI compiles scripts through a generated Scala file underneath the project root.
// CHECK: public module ProjectRootModule : @[[[FILE:\.scala-build/[^ ]+/src_generated/main/SourceInfoProjectRoot.scala]] {{[0-9]+}}:{{[0-9]+}}]
// CHECK-NEXT: input clock : Clock @[[[FILE]] {{[0-9]+}}:{{[0-9]+}}]
class ProjectRootModule extends Module {
  // The IO locator comes from the SourceInfo macro.
  // CHECK: output out : UInt<1> @[[[FILE]] {{[0-9]+}}:{{[0-9]+}}]
  val out = IO(Output(Bool()))
  // CHECK: connect out, UInt<1>(0h1) @[[[FILE]] {{[0-9]+}}:{{[0-9]+}}]
  out := true.B
}

println(circt.stage.ChiselStage.emitCHIRRTL(new ProjectRootModule))

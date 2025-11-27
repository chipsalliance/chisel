// RUN: scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" %s -- chirrtl | FileCheck %s -check-prefix=FIRRTL
// RUN: scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" %s -- verilog | FileCheck %s -check-prefix=VERILOG
// RUN: scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" %s -- chirrtl | firtool -format=fir  | FileCheck %s -check-prefix=VERILOG

import chisel3._

class FooBundle extends Bundle {
  val foo = Input(UInt(3.W))
}

// FIRRTL-LABEL: circuit FooModule :
// FIRRTL:         public module FooModule :
// FIRRTL-NEXT:      input clock : Clock
// FIRRTL-NEXT:      input reset : UInt<1>
// FIRRTL-NEXT:      output io : { flip foo : UInt<3>}
// FIRRTL:           skip

// VERILOG-LABEL: module FooModule(
// VERILOG-NEXT:    input clock,
// VERILOG-NEXT:          reset,
// VERILOG-NEXT:    input [2:0] io_foo
// VERILOG-NEXT:  );

class FooModule extends Module {
  val io = IO(new FooBundle)
}

args.head match {
  case "chirrtl" => {
    println(circt.stage.ChiselStage.emitCHIRRTL(new FooModule))
  }
  case "verilog" => {
    println(circt.stage.ChiselStage.emitSystemVerilog(new FooModule))
  }
}

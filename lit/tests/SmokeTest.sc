// RUN: not scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION %s -- chirrtl 2>&1 | FileCheck %s -check-prefix=NO-COMPILER-PLUGIN
// RUN: scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" %s -- chirrtl | FileCheck %s -check-prefix=SFC-FIRRTL
// RUN: scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" %s -- chirrtl | firtool -format=fir  | FileCheck %s -check-prefix=VERILOG
// RUN: scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" --java-opt="--enable-native-access=ALL-UNNAMED" --java-opt="--enable-preview" --java-opt="-Djava.library.path=%JAVALIBRARYPATH" %s -- panama-firrtl | FileCheck %s -check-prefix=MFC-FIRRTL
// RUN: scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" --java-opt="--enable-native-access=ALL-UNNAMED" --java-opt="--enable-preview" --java-opt="-Djava.library.path=%JAVALIBRARYPATH" %s -- panama-verilog | FileCheck %s -check-prefix=VERILOG

import chisel3._

class FooBundle extends Bundle {
  val foo = Input(UInt(3.W))
}

// SFC-FIRRTL-LABEL: circuit FooModule :
// SFC-FIRRTL-NEXT:    layer Verification, bind, "Verification" :
// SFC-FIRRTL-NEXT:      layer Assert, bind, "Verification/Assert" :
// SFC-FIRRTL-NEXT:      layer Assume, bind, "Verification/Assume" :
// SFC-FIRRTL-NEXT:      layer Cover, bind, "Verification/Cover" :
// SFC-FIRRTL-NEXT:    public module FooModule :
// SFC-FIRRTL-NEXT:      input clock : Clock
// SFC-FIRRTL-NEXT:      input reset : UInt<1>
// SFC-FIRRTL-NEXT:      output io : { flip foo : UInt<3>}
// SFC-FIRRTL:           skip

// MFC-FIRRTL-LABEL: circuit FooModule :
// MFC-FIRRTL-NEXT:    public module FooModule :
// MFC-FIRRTL-NEXT:      input clock : Clock
// MFC-FIRRTL-NEXT:      input reset : UInt<1>
// MFC-FIRRTL-NEXT:      output io : { flip foo : UInt<3> }

// VERILOG-LABEL: module FooModule(
// VERILOG-NEXT:    input clock,
// VERILOG-NEXT:          reset,
// VERILOG-NEXT:    input [2:0] io_foo
// VERILOG-NEXT:  );

// NO-COMPILER-PLUGIN-LABEL: assertion failed: The Chisel compiler plugin is now required for compiling Chisel code.

class FooModule extends Module {
  val io = IO(new FooBundle)
}

args.head match {
  case "chirrtl" => {
    println(circt.stage.ChiselStage.emitCHIRRTL(new FooModule))
  }
  case "panama-firrtl" => {
    println(lit.utility.panamaconverter.firrtlString(new FooModule))
  }
  case "panama-verilog" => {
    println(lit.utility.panamaconverter.verilogString(new FooModule))
  }
}

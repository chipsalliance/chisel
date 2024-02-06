// RUN: scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" %s -- chirrtl | FileCheck %s -check-prefix=SFC-FIRRTL
// RUN: scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" --java-opt="--enable-native-access=ALL-UNNAMED" --java-opt="--enable-preview" --java-opt="-Djava.library.path=%JAVALIBRARYPATH" %s -- panama-om | FileCheck %s -check-prefix=MFC-OM

import chisel3._
import chisel3.properties._
import chisel3.panamaom._
import lit.utility._

// SFC-FIRRTL-LABEL: circuit IntPropTest :
// SFC-FIRRTL-NEXT: module IntPropTest :
// SFC-FIRRTL-NEXT: output intProp : Integer

// MFC-OM-LABEL: circuit IntPropTest :
// MFC-OM-NEXT: module IntPropTest :
// MFC-OM-NEXT: output intProp : Integer
class IntPropTest extends RawModule {
  val intProp = IO(Output(Property[Int]()))
  intProp := Property(1)
}

args.head match {
  case "chirrtl" =>
    println(circt.stage.ChiselStage.emitCHIRRTL(new IntPropTest))
  case "panama-om" =>
    println(lit.utility.panamaconverter.firrtlString(new IntPropTest))
  case _ =>
}

class PropertyTest extends Module {
  val i = IO(Input(UInt(8.W)))
  val o = IO(Output(UInt(8.W)))

  val m = Module(new Module {
    val i = IO(Input(UInt(8.W)))
    val r = RegNext(i)
    val o = IO(Output(UInt(8.W)))
    val p = IO(Output(Property[Int]()))
    p := Property(789)
    o := r
    val nested = Module(new Module {
      val i = IO(Input(UInt(8.W)))
      val r = RegNext(i)
      val o = IO(Output(UInt(8.W)))
      val p = IO(Output(Property[Int]()))
      p := Property(789)
      o := r
    })
    nested.i := i
    o := nested.o
  })
  m.i := i
  o := m.o

  val p = IO(Output(Property[Path]()))
  p := Property(Path(i))

  val a = IO(Output(Property[Seq[Seq[Property[Int]]]]()))
  val b = IO(Output(Property[Seq[Property[Seq[Int]]]]()))
  a := Property(Seq[Seq[Int]](Seq(123)))
  b := Property(Seq[Seq[Int]](Seq(456)))
}

args.head match {
  case "panama-om" =>
    val converter = lit.utility.panamaconverter.getConverter(new PropertyTest)
    lit.utility.panamaconverter.runAllPass(converter)

    val om = converter.om()
    val evaluator = om.evaluator()
    val obj = evaluator.instantiate("PropertyTest_Class", Seq(om.newBasePathEmpty))

    // CHECK: OMReferenceTarget:~PropertyTest|PropertyTest>i
    println(obj.field("p").asInstanceOf[PanamaCIRCTOMEvaluatorValuePath].asString)

    // CHECK:      .a => { [ [ prim{omInteger{123}} ] ] }
    // CHECK-NEXT: .b => { [ [ prim{omInteger{456}} ] ] }
    // CHECK-NEXT: .p => { path{OMReferenceTarget:~PropertyTest|PropertyTest>i} }
    obj.foreachField((name, value) => println(s".$name => { ${value.display} }"))

    // CHECK:      module{_1_Anon}
    // CHECK-NEXT: module{PropertyTest_Anon}
    // CHECK-NEXT: module{PropertyTest}
    converter.foreachHwModule(name => println(s"module{$name}"))
}

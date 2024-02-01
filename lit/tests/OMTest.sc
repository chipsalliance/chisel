// RUN: scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" --java-opt="--enable-native-access=ALL-UNNAMED" --java-opt="--enable-preview" --java-opt="-Djava.library.path=%JAVALIBRARYPATH" %s -- panama-omstring | FileCheck %s

import chisel3._
import chisel3.properties._
import chisel3.panamaom._
import chisel3.panamaconverter._

class PropertyTest extends RawModule {
  val i = IO(Input(UInt(8.W)))

  val p = IO(Output(Property[Path]()))
  p := Property(Path(i))

  val a = IO(Output(Property[Seq[Seq[Property[Int]]]]()))
  val b = IO(Output(Property[Seq[Property[Seq[Int]]]]()))
  a := Property(Seq[Seq[Int]](Seq(123)))
  b := Property(Seq[Seq[Int]](Seq(456)))
}

args.head match {
  case "panama-omstring" =>
    val converter: PanamaCIRCTConverter = lit.utility.panamaconverter.getConverter(new PropertyTest)
    val pm = converter.passManager()
    assert(pm.populatePreprocessTransforms())
    assert(pm.populateCHIRRTLToLowFIRRTL())
    assert(pm.populateLowFIRRTLToHW())
    assert(pm.populateFinalizeIR())
    assert(pm.run())

    val om = converter.om()
    val evaluator = om.evaluator()
    val obj = evaluator.instantiate("PropertyTest_Class", Seq(om.newBasePathEmpty))

    // CHECK: OMReferenceTarget:~PropertyTest|PropertyTest>i
    println(obj.field("p").asInstanceOf[PanamaCIRCTOMEvaluatorValuePath].asString)

    // CHECK:      .a => { [ [ prim{omInteger{123}} ] ] }
    // CHECK-NEXT: .b => { [ [ prim{omInteger{456}} ] ] }
    // CHECK-NEXT: .p => { path{OMReferenceTarget:~PropertyTest|PropertyTest>i} }
    obj.foreachField((name, value) => println(s".$name => { ${value.display} }"))
}

// RUN: scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" --scala-option="-Ymacro-annotations" --java-opt="--enable-native-access=ALL-UNNAMED" --java-opt="--enable-preview" --java-opt="-Djava.library.path=%JAVALIBRARYPATH" %s -- chirrtl | firtool --format=fir --ir-verilog | scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" --scala-option="-Ymacro-annotations" --java-opt="--enable-native-access=ALL-UNNAMED" --java-opt="--enable-preview" --java-opt="-Djava.library.path=%JAVALIBRARYPATH" %s -- mlir-verilog | FileCheck %s

import scala.io

import chisel3._
import chisel3.properties._
import chisel3.panamaconverter.PanamaCIRCTConverter
import chisel3.panamaom._
import chisel3.experimental.hierarchy.{instantiable, public, Definition, Instance}

import lit.utility._

// An abstract description of a CSR, represented as a Class.
@instantiable
class CSRDescription extends Class {
  // An output Property indicating the CSR name.
  val identifier = IO(Output(Property[String]()))
  // An output Property describing the CSR.
  val description = IO(Output(Property[String]()))
  // An output Property indicating the CSR width.
  val width = IO(Output(Property[Int]()))

  // Input Properties to be passed to Objects representing instances of the Class.
  @public val identifierIn = IO(Input(Property[String]()))
  @public val descriptionIn = IO(Input(Property[String]()))
  @public val widthIn = IO(Input(Property[Int]()))

  // Simply connect the inputs to the outputs to expose the values.
  identifier := identifierIn
  description := descriptionIn
  width := widthIn
}

// A hardware module representing a CSR and its description.
class CSRModule(
  csrDescDef:     Definition[CSRDescription],
  width:          Int,
  identifierStr:  String,
  descriptionStr: String)
    extends Module {
  override def desiredName = identifierStr

  // Create a hardware port for the CSR value.
  val value = IO(Output(UInt(width.W)))

  // Create a property port for a reference to the CSR description object.
  val description = IO(Output(csrDescDef.getPropertyType))

  // Instantiate a CSR description object, and connect its input properties.
  val csrDescription = Instance(csrDescDef)
  csrDescription.identifierIn := Property(identifierStr)
  csrDescription.descriptionIn := Property(descriptionStr)
  csrDescription.widthIn := Property(width)

  // Create a register for the hardware CSR. A real implementation would be more involved.
  val csr = RegInit(0.U(width.W))

  // Assign the CSR value to the hardware port.
  value := csr

  // Assign a reference to the CSR description object to the property port.
  description := csrDescription.getPropertyReference
}

// The entrypoint module.
class Top extends Module {
  // Create a Definition for the CSRDescription Class.
  val csrDescDef = Definition(new CSRDescription)

  // Get the CSRDescription ClassType.
  val csrDescType = csrDescDef.getClassType

  // Create a property port to collect all the CSRDescription object references.
  val descriptions = IO(Output(Property[Seq[csrDescType.Type]]()))

  // Instantiate a couple CSR modules.
  val mcycle = Module(new CSRModule(csrDescDef, 64, "mcycle", "Machine cycle counter."))
  val minstret = Module(new CSRModule(csrDescDef, 64, "minstret", "Machine instructions-retired counter."))

  // Assign references to the CSR description objects to the property port.
  descriptions := Property(Seq(mcycle.description.as(csrDescType), minstret.description.as(csrDescType)))
}

args.head match {
  case "chirrtl" =>
    println(circt.stage.ChiselStage.emitCHIRRTL(new Top))
  case "mlir-verilog" =>
    val mlir = Iterator.continually(io.StdIn.readLine).takeWhile(_ != null).mkString("\n")
    val converter = PanamaCIRCTConverter.newWithMlir(mlir)
    val pm = converter.passManager()
    assert(pm.populateFinalizeIR())
    assert(pm.run())

    val om = converter.om()
    val evaluator = om.evaluator()

    val top = evaluator.instantiate("Top_Class", Seq(om.newBasePathEmpty)).get

    // CHECK: .descriptions => { [ obj{.description => { Machine cycle counter. }, .identifier => { mcycle }, .width => { 64 }}, obj{.description => { Machine instructions-retired counter. }, .identifier => { minstret }, .width => { 64 }} ] }
    top.foreachField((name, value) => println(s".$name => { ${value.toString} }"))
  case _ =>
}

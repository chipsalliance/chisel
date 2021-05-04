// SPDX-License-Identifier: Apache-2.0

package chisel3.stage

import firrtl.annotations.{Annotation, NoTargetAnnotation}
import firrtl.options.{CustomFileEmission, HasShellOptions, OptionsException, ShellOption, StageOptions, Unserializable}
import firrtl.options.Viewer.view
import chisel3.{ChiselException, Module}
import chisel3.RawModule
import chisel3.internal.Builder
import chisel3.internal.firrtl.{Circuit, Emitter => OldEmitter}
import firrtl.AnnotationSeq
import java.io.File

/** Mixin that indicates that this is an [[firrtl.annotations.Annotation]] used to generate a [[ChiselOptions]] view.
  */
sealed trait ChiselOption { this: Annotation => }

/** Disable the execution of the FIRRTL compiler by Chisel
  */
case object NoRunFirrtlCompilerAnnotation
    extends NoTargetAnnotation
    with ChiselOption
    with HasShellOptions
    with Unserializable {

  val options = Seq(
    new ShellOption[Unit](
      longOption = "no-run-firrtl",
      toAnnotationSeq = _ => Seq(NoRunFirrtlCompilerAnnotation),
      helpText = "Do not run the FIRRTL compiler (generate FIRRTL IR from Chisel and exit)",
      shortOption = Some("chnrf") ) )

}

/** On an exception, this will cause the full stack trace to be printed as opposed to a pruned stack trace.
  */
case object PrintFullStackTraceAnnotation
    extends NoTargetAnnotation
    with ChiselOption
    with HasShellOptions
    with Unserializable {

  val options = Seq(
    new ShellOption[Unit](
      longOption = "full-stacktrace",
      toAnnotationSeq = _ => Seq(PrintFullStackTraceAnnotation),
      helpText = "Show full stack trace when an exception is thrown" ) )

}

/** An [[firrtl.annotations.Annotation]] storing a function that returns a Chisel module
  * @param gen a generator function
  */
case class ChiselGeneratorAnnotation(gen: () => RawModule) extends NoTargetAnnotation with Unserializable {

  /** Run elaboration on the Chisel module generator function stored by this [[firrtl.annotations.Annotation]]
    */
  def elaborate: AnnotationSeq  = {
    val (circuit, dut) = Builder.build(Module(gen()))
    Seq(ChiselCircuitAnnotation(circuit), DesignAnnotation(dut))
  }

}

object ChiselGeneratorAnnotation extends HasShellOptions {

  /** Construct a [[ChiselGeneratorAnnotation]] with a generator function that will try to construct a Chisel Module
    * from using that Module's name. The Module must both exist in the class path and not take parameters.
    * @param name a module name
    * @throws firrtl.options.OptionsException if the module name is not found or if no parameterless constructor for
    * that Module is found
    */
  def apply(name: String): ChiselGeneratorAnnotation = {
    val gen = () => try {
      Class.forName(name).asInstanceOf[Class[_ <: RawModule]].newInstance()
    } catch {
      case e: ClassNotFoundException =>
        throw new OptionsException(s"Unable to locate module '$name'! (Did you misspell it?)", e)
      case e: InstantiationException =>
        throw new OptionsException(
          s"Unable to create instance of module '$name'! (Does this class take parameters?)", e)
    }
    ChiselGeneratorAnnotation(gen)
  }

  val options = Seq(
    new ShellOption[String](
      longOption = "module",
      toAnnotationSeq = (a: String) => Seq(ChiselGeneratorAnnotation(a)),
      helpText = "The name of a Chisel module to elaborate (module must be in the classpath)",
      helpValueName = Some("<package>.<module>") ) )

}

/** Stores a Chisel Circuit
  * @param circuit a Chisel Circuit
  */
case class ChiselCircuitAnnotation(circuit: Circuit)
    extends NoTargetAnnotation
    with ChiselOption
    with Unserializable {
  /* Caching the hashCode for a large circuit is necessary due to repeated queries.
   * Not caching the hashCode will cause severe performance degredations for large [[Circuit]]s.
   */
  override lazy val hashCode: Int = circuit.hashCode
}

object CircuitSerializationAnnotation {
  sealed trait Format {
    def extension: String
  }
  case object FirrtlFileFormat extends Format {
    def extension = ".fir"
  }
  case object ProtoBufFileFormat extends Format {
    def extension = ".pb"
  }
}

import CircuitSerializationAnnotation._

/** Wraps a [[Circuit]] for serialization via [[CustomFileEmission]]
  * @param circuit a Chisel Circuit
  * @param filename name of destination file (excludes file extension)
  * @param format serialization file format (sets file extension)
  */
case class CircuitSerializationAnnotation(circuit: Circuit, filename: String, format: Format)
    extends NoTargetAnnotation
    with CustomFileEmission {
  /* Caching the hashCode for a large circuit is necessary due to repeated queries.
   * Not caching the hashCode will cause severe performance degredations for large [[Circuit]]s.
   */
  override lazy val hashCode: Int = circuit.hashCode

  protected def baseFileName(annotations: AnnotationSeq): String = filename

  protected def suffix: Option[String] = Some(format.extension)

  // TODO Use lazy Iterables so that we don't have to materialize full intermediate data structures
  override def getBytes: Iterable[Byte] = format match {
    case FirrtlFileFormat => OldEmitter.emit(circuit).getBytes
    case ProtoBufFileFormat =>
      val ostream = new java.io.ByteArrayOutputStream
      val modules = circuit.components.map(m => () => chisel3.internal.firrtl.Converter.convert(m))
      firrtl.proto.ToProto.writeToStreamFast(ostream, firrtl.ir.NoInfo, modules, circuit.name)
      ostream.toByteArray
  }
}

case class ChiselOutputFileAnnotation(file: String) extends NoTargetAnnotation with ChiselOption with Unserializable

object ChiselOutputFileAnnotation extends HasShellOptions {

  val options = Seq(
    new ShellOption[String](
      longOption = "chisel-output-file",
      toAnnotationSeq = (a: String) => Seq(ChiselOutputFileAnnotation(a)),
      helpText = "Write Chisel-generated FIRRTL to this file (default: <circuit-main>.fir)",
      helpValueName = Some("<file>") ) )

}

/** Contains the top-level elaborated Chisel design.
  *
  * By default is created during Chisel elaboration and passed to the FIRRTL compiler.
  * @param design top-level Chisel design
  * @tparam DUT Type of the top-level Chisel design
  */
case class DesignAnnotation[DUT <: RawModule](design: DUT) extends NoTargetAnnotation with Unserializable

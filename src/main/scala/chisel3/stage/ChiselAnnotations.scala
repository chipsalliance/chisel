// See LICENSE for license details.

package chisel3.stage

import firrtl.AnnotationSeq
import firrtl.annotations.{Annotation, NoTargetAnnotation}
import firrtl.options.{HasScoptOptions, OptionsException, Unserializable}

import chisel3.{ChiselException, Module}
import chisel3.experimental.RawModule
import chisel3.internal.Builder
import chisel3.internal.firrtl.Circuit

import scopt.OptionParser

/** Mixin that indicates that this is an [[firrtl.annotations.Annotation Annotation]] used to generate a
  * [[ChiselOptions]] view.
  */
sealed trait ChiselOption extends Unserializable { this: Annotation => }

/** Disable the execution of the FIRRTL compiler by Chisel
  */
case object NoRunFirrtlCompilerAnnotation extends NoTargetAnnotation with ChiselOption with HasScoptOptions {

  def addOptions(p: OptionParser[AnnotationSeq]): Unit = p
    .opt[Unit]("no-run-firrtl")
    .abbr("chnrf")
    .action( (x, c) => NoRunFirrtlCompilerAnnotation +: c )
    .unbounded()
    .text("Stop after chisel emits chirrtl file")

}

/** On an exception, this will cause the full stack trace to be printed as opposed to a pruned stack trace.
  */
case object PrintFullStackTraceAnnotation extends NoTargetAnnotation with ChiselOption with HasScoptOptions {

  def addOptions(p: OptionParser[AnnotationSeq]): Unit = p
    .opt[Unit]("full-stacktrace")
    .action( (x, c) => PrintFullStackTraceAnnotation +: c )
    .unbounded()
    .text("Do not trim stack trace")

}

/** An [[firrtl.annotations.Annotation Annotation]] storing a function that returns a Chisel module
  * @param gen a generator function
  */
case class ChiselGeneratorAnnotation(gen: () => RawModule) extends NoTargetAnnotation with Unserializable {

  /** Run elaboration on the Chisel module generator function stored by this [[firrtl.annotations.Annotation Annotation]]
    */
  def elaborate: ChiselCircuitAnnotation = try {
    ChiselCircuitAnnotation(Builder.build(Module(gen())))
  } catch {
    case e @ (_: OptionsException | _: ChiselException) => throw e
    case e: Throwable =>
      throw new OptionsException(s"Exception thrown when elaborating ChiselGeneratorAnnotation", e)
  }

}

object ChiselGeneratorAnnotation extends HasScoptOptions {

  /** Construct a [[ChiselGeneratorAnnotation]] with a generator function that will try to construct a Chisel Module from
    * using that Module's name. The Module must both exist in the class path and not take parameters.
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
        throw new OptionsException(s"Unable to create instance of module '$name'! (Does this class take parameters?)", e)
    }
    ChiselGeneratorAnnotation(gen)
  }

  def addOptions(p: OptionParser[AnnotationSeq]): Unit = p
    .opt[String]("module")
    .action{ (x, c) => ChiselGeneratorAnnotation(x) +: c }
    .unbounded()
    .text("The name of a Chisel module (in the classpath) to elaborate")

}

/** Stores a Chisel [[chisel3.internal.firrtl.Circuit Circuit]]
  * @param circuit a Chisel Circuit
  */
case class ChiselCircuitAnnotation(circuit: Circuit) extends NoTargetAnnotation with ChiselOption

case class ChiselOutputFileAnnotation(file: String) extends NoTargetAnnotation with ChiselOption

object ChiselOutputFileAnnotation extends HasScoptOptions {

  def addOptions(p: OptionParser[AnnotationSeq]): Unit = p
    .opt[String]("chisel-output-file")
    .valueName("FILE")
    .action( (x, c) => ChiselOutputFileAnnotation(x) +: c )
    .unbounded()
    .text("sets an output file name for the Chisel-generated FIRRTL circuit")

}

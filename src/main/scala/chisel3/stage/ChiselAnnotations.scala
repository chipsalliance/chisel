// See LICENSE for license details.

package chisel3.stage

import java.io.{File, FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}

import firrtl.annotations.{Annotation, NoTargetAnnotation}
import firrtl.options.{HasShellOptions, OptionsException, ShellOption, Unserializable}
import chisel3.{ChiselException, Module}
import chisel3.RawModule
import chisel3.experimental.BaseModule
import chisel3.internal.{Builder, ChiselCache, ChiselCacheTag}
import chisel3.internal.firrtl.Circuit
import com.twitter.chill.MeatLocker
import firrtl.AnnotationSeq

/** Mixin that indicates that this is an [[firrtl.annotations.Annotation]] used to generate a [[ChiselOptions]] view.
  */
sealed trait ChiselOption extends Unserializable { this: Annotation => }

/** Disable the execution of the FIRRTL compiler by Chisel
  */
case object NoRunFirrtlCompilerAnnotation extends NoTargetAnnotation with ChiselOption with HasShellOptions {

  val options = Seq(
    new ShellOption[Unit](
      longOption = "no-run-firrtl",
      toAnnotationSeq = _ => Seq(NoRunFirrtlCompilerAnnotation),
      helpText = "Do not run the FIRRTL compiler (generate FIRRTL IR from Chisel and exit)",
      shortOption = Some("chnrf") ) )

}

/** On an exception, this will cause the full stack trace to be printed as opposed to a pruned stack trace.
  */
case object PrintFullStackTraceAnnotation extends NoTargetAnnotation with ChiselOption with HasShellOptions {

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
  def elaborate: AnnotationSeq =  {
    try {
      val (circuit, dut) = Builder.build(Module(gen()))
      Seq(ChiselCircuitAnnotation(circuit), DesignAnnotation(dut))
    } catch {
      case e @ (_: OptionsException | _: ChiselException) => throw e
      case e: Throwable =>
        throw new OptionsException(s"Exception thrown when elaborating ChiselGeneratorAnnotation", e)
    }
  }

  def elaborateWithCache(cache: ChiselCacheAnnotation): AnnotationSeq  = {
    try {
      println("HERE2")
      val (circuit, dut, resCache) = Builder.buildWithCache(cache.getCache, Module(gen()))
      cache.writeCache(resCache)
      Seq(ChiselCircuitAnnotation(circuit), DesignAnnotation(dut))
    } catch {
      case e @ (_: OptionsException | _: ChiselException) => throw e
      case e: Throwable =>
        throw new OptionsException(s"Exception thrown when elaborating ChiselGeneratorAnnotation", e)
    }
  }

  def reload: RawModule = {
    val dut = Builder.reload(gen())
    dut
  }
}

object ChiselCacheAnnotation extends HasShellOptions {

  val options = Seq(
    new ShellOption[String](
      longOption = "cacheInputDirectory",
      toAnnotationSeq = (a: String) => Seq(ChiselCacheAnnotation(Some(a), None)),
      helpText = "The relative or absolute path to the directory containing the cached Chisel elaborations to consume.",
      helpValueName = Some("<path>") ) ,
  new ShellOption[String](
    longOption = "cacheOutputDirectory",
    toAnnotationSeq = (a: String) => Seq(ChiselCacheAnnotation(None, Some(a))),
    helpText = "The relative or absolute path to the directory to write cached Chisel elaborations",
    helpValueName = Some("<path>") ) )
}

/** Construct a [[ChiselCacheAnnotation]] with a path to the cache directory
  * @param dir a module name
  * @throws firrtl.options.OptionsException if the module name is not found or if no parameterless constructor for
  * that Module is found
  */
case class ChiselCacheAnnotation(inputDir: Option[String], outputDir: Option[String]) extends NoTargetAnnotation {

  def getListOfFiles(dir: String):List[File] = {
    val d = new File(dir)
    if (d.exists && d.isDirectory) {
      d.listFiles.filter(_.isFile).toList
    } else {
      List[File]()
    }
  }

  def getCache: ChiselCache = {
    if(inputDir.nonEmpty) {
      val dir = inputDir.get
      val files = getListOfFiles(dir)
      ChiselCache(files.collect {
        case f: File if f.getName.split('.').length == 2 && f.getName.split('.').last == "cache" =>
          val tag = ChiselCacheTag(Integer.parseInt(f.getName.split('.').head, 16))
          val module = ChiselGeneratorAnnotation(() => {
            val ois = new ObjectInputStream(new FileInputStream(f.getAbsolutePath))
            val obj = ois.readObject.asInstanceOf[MeatLocker[RawModule]]
            ois.close()
            obj.get
          }).reload
          (tag, module)
      }.toMap)
    } else ChiselCache(Map.empty[ChiselCacheTag, RawModule])
  }

  def writeCache(cache: ChiselCache): Unit = {
    if(outputDir.nonEmpty) {
      val dir = outputDir.get
      cache.cache.foreach {
        case (tag, m) =>
          val oos = new ObjectOutputStream(new FileOutputStream(dir + tag.filename))
          oos.writeObject(MeatLocker(m))
          oos.close
      }
    }
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
case class ChiselCircuitAnnotation(circuit: Circuit) extends NoTargetAnnotation with ChiselOption

case class ChiselOutputFileAnnotation(file: String) extends NoTargetAnnotation with ChiselOption

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

case class StashAnnotation()
// See LICENSE for license details.

package Chisel

import collection.mutable.{ArrayBuffer, HashSet, HashMap, Stack, LinkedHashSet, Queue => ScalaQueue}
import scala.math.min

trait FileSystemUtilities {
  def createOutputFile(name: String, contents: String) {
    val f = new java.io.FileWriter(name)
    f.write(contents)
    f.close
  }
}

object Driver extends FileSystemUtilities {

  /** Instantiates a ChiselConfig class with the given name and uses it for elaboration */
  def elaborateWithConfigName[T <: Module](
      gen: () => T,
      configClassName: String,
      projectName: Option[String] = None,
      collectConstraints: Boolean = false): Unit = {
    val className = projectName match {
      case Some(pn) => s"$pn.$configClassName"
      case None => configClassName
    }
    val config = try {
      Class.forName(className).newInstance.asInstanceOf[ChiselConfig]
    } catch {
      case e: java.lang.ClassNotFoundException =>
        throwException("Could not find the ChiselConfig subclass you asked for (i.e. \"" +
                          className + "\"), did you misspell it?", e)
    }
    elaborateWithConfig(gen, config, collectConstraints)
  }

  /** Uses the provided ChiselConfig for elaboration */
  def elaborateWithConfig[T <: Module](
      gen: () => T,
      config: ChiselConfig,
      collectConstraints: Boolean = false): Unit = {
    val world = if(collectConstraints) config.toCollector else config.toInstance
    val p = Parameters.root(world)
    config.topConstraints.foreach(c => p.constrain(c))
    elaborate(gen, p, config)
  }

  /** Elaborates the circuit specified in the gen function, optionally uses
    * a parameter space to supply context-aware values.
    *  TODO: Distinguish between cases where we dump to file vs return IR for
    *        use by other Drivers.
    */
  private[Chisel] def elaborateWrappedModule[T <: Module](gen: () => T, p: Parameters, c: Option[ChiselConfig]) {
    val ir = Builder.build(gen())
    val name = c match {
      case None => ir.name
      case Some(config) => s"${ir.name}.$config"
    }
    createOutputFile(s"$name.knb", p.getKnobs)
    createOutputFile(s"$name.cst", p.getConstraints)
    createOutputFile(s"$name.prm", ir.parameterDump.getDump)
    createOutputFile(s"$name.fir", ir.emit)
  }
  def elaborate[T <: Module](gen: () => T): Unit =
    elaborate(gen, Parameters.empty)
  def elaborate[T <: Module](gen: () => T, p: Parameters): Unit =
    elaborateWrappedModule(() => Module(gen())(p), p, None)
  private def elaborate[T <: Module](gen: () => T, p: Parameters, c: ChiselConfig): Unit =
    elaborateWrappedModule(() => Module(gen())(p), p, Some(c))
}

object chiselMain {
  def apply[T <: Module](args: Array[String], gen: () => T, p: Parameters = Parameters.empty): Unit =
    Driver.elaborateWrappedModule(gen, p, None)
}

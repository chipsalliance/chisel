/*
 Copyright (c) 2011, 2012, 2013, 2014 The Regents of the University of
 California (Regents). All Rights Reserved.  Redistribution and use in
 source and binary forms, with or without modification, are permitted
 provided that the following conditions are met:

    * Redistributions of source code must retain the above
      copyright notice, this list of conditions and the following
      two paragraphs of disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      two paragraphs of disclaimer in the documentation and/or other materials
      provided with the distribution.
    * Neither the name of the Regents nor the names of its contributors
      may be used to endorse or promote products derived from this
      software without specific prior written permission.

 IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
 SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
 ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF
 REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF
 ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION
 TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
 MODIFICATIONS.
*/

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

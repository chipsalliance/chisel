// See LICENSE for license details.

package Chisel

import scala.sys.process._


trait FileSystemUtilities {
  def createOutputFile(name: String, contents: String): String = {
    val f = new java.io.FileWriter(name)
    f.write(contents)
    f.close
    name
  }
}

trait BackendCompilationUtilities {

}

object Driver extends FileSystemUtilities with BackendCompilationUtilities {

  /** Elaborates the Module specified in the gen function into a Circuit 
    *
    *  @param gen a function that creates a Module hierarchy
    *
    *  @return the resulting Chisel IR in the form of a Circuit
    */
  def elaborate[T <: Module](gen: () => T): Circuit = Builder.build(Module(gen()))
  
  def emit[T <: Module](gen: () => T): String = elaborate(gen).emit

  def dumpFirrtl(ir: Circuit, optName: Option[String]): String = {
    val name = optName.getOrElse(ir.name)
    createOutputFile(s"$name.fir", ir.emit)
  }
}

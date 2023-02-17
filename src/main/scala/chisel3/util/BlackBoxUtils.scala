// SPDX-License-Identifier: Apache-2.0

package chisel3.util

import chisel3._
import chisel3.experimental.ChiselAnnotation
import firrtl.transforms.{BlackBoxInlineAnno, BlackBoxNotFoundException, BlackBoxPathAnno}
import firrtl.annotations.ModuleName
import logger.LazyLogging

private[util] object BlackBoxHelpers {

  implicit class BlackBoxInlineAnnoHelpers(anno: BlackBoxInlineAnno.type) extends LazyLogging {

    /** Generate a BlackBoxInlineAnno from a Java Resource and a module name. */
    def fromResource(resourceName: String, moduleName: ModuleName) = try {
      val blackBoxFile = os.resource / os.RelPath(resourceName.dropWhile(_ == '/'))
      val contents = os.read(blackBoxFile)
      if (contents.size > BigInt(2).pow(20)) {
        val message =
          s"Black box resource $resourceName, which will be converted to an inline annotation, is greater than 1 MiB." +
            "This may affect compiler performance. Consider including this resource via a black box path."
        logger.warn(message)
      }
      BlackBoxInlineAnno(moduleName, blackBoxFile.last, contents)
    } catch {
      case e: os.ResourceNotFoundException =>
        throw new BlackBoxNotFoundException(resourceName, e.getMessage)
    }
  }
}

import BlackBoxHelpers._

trait HasBlackBoxResource extends BlackBox {
  self: BlackBox =>

  /** Copies a Java resource containing some text into the output directory. This is typically used to copy a Verilog file
    * to the final output directory, but may be used to copy any Java resource (e.g., a C++ testbench).
    *
    * Resource files are located in project_root/src/main/resources/.
    * Example of adding the resource file project_root/src/main/resources/blackbox.v:
    * {{{
    * addResource("/blackbox.v")
    * }}}
    */
  def addResource(blackBoxResource: String): Unit = {
    val anno = new ChiselAnnotation {
      def toFirrtl = BlackBoxInlineAnno.fromResource(blackBoxResource, self.toNamed)
    }
    chisel3.experimental.annotate(anno)
  }
}

trait HasBlackBoxInline extends BlackBox {
  self: BlackBox =>

  /** Creates a black box verilog file, from the contents of a local string
    *
    * @param blackBoxName   The black box module name, to create filename
    * @param blackBoxInline The black box contents
    */
  def setInline(blackBoxName: String, blackBoxInline: String): Unit = {
    val anno = new ChiselAnnotation {
      def toFirrtl = BlackBoxInlineAnno(self.toNamed, blackBoxName, blackBoxInline)
    }
    chisel3.experimental.annotate(anno)
  }
}

trait HasBlackBoxPath extends BlackBox {
  self: BlackBox =>

  /** Copies a file to the target directory
    *
    * This works with absolute and relative paths. Relative paths are relative
    * to the current working directory, which is generally not the same as the
    * target directory.
    */
  def addPath(blackBoxPath: String): Unit = {
    val anno = new ChiselAnnotation {
      def toFirrtl = BlackBoxPathAnno(self.toNamed, blackBoxPath)
    }
    chisel3.experimental.annotate(anno)
  }
}

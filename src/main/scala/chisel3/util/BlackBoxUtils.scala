// SPDX-License-Identifier: Apache-2.0

package chisel3.util

import chisel3._
import chisel3.experimental.{ChiselAnnotation, RunFirrtlTransform}
import firrtl.transforms.{BlackBoxPathAnno, BlackBoxResourceAnno, BlackBoxInlineAnno, BlackBoxSourceHelper,
  BlackBoxNotFoundException}

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
    val anno = new ChiselAnnotation with RunFirrtlTransform {
      def toFirrtl = try {
        val blackBoxFile = os.resource / os.RelPath(blackBoxResource.dropWhile(_ == '/'))
        BlackBoxInlineAnno(self.toNamed, blackBoxFile.last, os.read(blackBoxFile))
      } catch {
        case e: os.ResourceNotFoundException =>
          throw new BlackBoxNotFoundException(blackBoxResource, e.getMessage)
      }
      def transformClass = classOf[BlackBoxSourceHelper]
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
    val anno = new ChiselAnnotation with RunFirrtlTransform {
      def toFirrtl = BlackBoxInlineAnno(self.toNamed, blackBoxName, blackBoxInline)
      def transformClass = classOf[BlackBoxSourceHelper]
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
    val anno = new ChiselAnnotation with RunFirrtlTransform {
      def toFirrtl = BlackBoxPathAnno(self.toNamed, blackBoxPath)
      def transformClass = classOf[BlackBoxSourceHelper]
    }
    chisel3.experimental.annotate(anno)
  }
}

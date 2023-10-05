// See LICENSE for license details.

package chisel3.util

import chisel3.experimental.{ChiselAnnotation, ExtModule}
import firrtl.transforms.{BlackBoxInlineAnno, BlackBoxNotFoundException, BlackBoxPathAnno}

import BlackBoxHelpers._

trait HasExtModuleResource extends ExtModule {
  self: ExtModule =>

  /** Copies a resource file to the target directory
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

trait HasExtModuleInline extends ExtModule {
  self: ExtModule =>

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

trait HasExtModulePath extends ExtModule {
  self: ExtModule =>

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

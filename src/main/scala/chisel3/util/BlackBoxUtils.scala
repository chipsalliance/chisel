// SPDX-License-Identifier: Apache-2.0

package chisel3.util

import chisel3._
import chisel3.experimental.BlackBoxHelpers.BlackBoxInlineAnnoHelpers
import firrtl.transforms.{BlackBoxInlineAnno, BlackBoxNotFoundException, BlackBoxPathAnno}

private object BlackBoxUtils {
  final val message =
    "this trait will be removed in Chisel 8, please switch from `BlackBox` to `ExtModule` which has the methods of this trait already available"
  final val since = "7.5.0"
}
import BlackBoxUtils._

@deprecated(message, since)
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
    chisel3.experimental.annotate(self)(Seq(BlackBoxInlineAnno.fromResource(blackBoxResource, self.toNamed)))
  }
}

@deprecated(message, since)
trait HasBlackBoxInline extends BlackBox {
  self: BlackBox =>

  /** Creates a black box verilog file, from the contents of a local string
    *
    * @param blackBoxName   The black box module name, to create filename
    * @param blackBoxInline The black box contents
    */
  def setInline(blackBoxName: String, blackBoxInline: String): Unit = {
    chisel3.experimental.annotate(self)(Seq(BlackBoxInlineAnno(self.toNamed, blackBoxName, blackBoxInline)))
  }
}

@deprecated(message, since)
trait HasBlackBoxPath extends BlackBox {
  self: BlackBox =>

  /** Copies a file to the target directory
    *
    * This works with absolute and relative paths. Relative paths are relative
    * to the current working directory, which is generally not the same as the
    * target directory.
    */
  def addPath(blackBoxPath: String): Unit = {
    chisel3.experimental.annotate(self)(Seq(BlackBoxPathAnno(self.toNamed, blackBoxPath)))
  }
}

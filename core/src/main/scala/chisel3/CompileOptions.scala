// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.language.experimental.macros
import scala.reflect.macros.blackbox.Context

trait CompileOptions { }

object CompileOptions {
  // Provides a low priority Strict default. Can be overridden by importing the NotStrict option.
  // Implemented as a macro to prevent this from being used inside chisel core.
  implicit def materialize: CompileOptions = macro materialize_impl

  def materialize_impl(c: Context): c.Tree = {
    import c.universe._
    q"_root_.chisel3.ExplicitCompileOptions.Strict"
  }
}

object ExplicitCompileOptions {

  case class CompileOptionsClass() extends CompileOptions

  // Collection of "not strict" connection compile options.
  // These provide compatibility with existing code.
  // Collection of "strict" connection compile options, preferred for new code.
  implicit val Strict = new CompileOptionsClass()
}

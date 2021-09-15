// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

import scala.language.experimental.macros
import scala.annotation.{StaticAnnotation, TypeConstraint}
import scala.reflect.macros.whitebox

private[chisel3] object adamMacro {

  def impl(c: whitebox.Context)(annottees: c.Expr[Any]*): c.Expr[Any] = {
    import c.universe._
    val result = {
      q"""
        ..$annottees
      """
    }
    
    c.error(c.enclosingPosition, "BLAH")
    c.Expr[Any](q"""""")
  }
}


class adam extends StaticAnnotation {
  def macroTransform(annottees: Any*): Any = macro adamMacro.impl
}
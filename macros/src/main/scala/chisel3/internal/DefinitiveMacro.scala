// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

import scala.language.experimental.macros
import scala.annotation.StaticAnnotation
import scala.reflect.macros.whitebox
//import scala.reflect.runtime.universe.WeakTypeTag

private[chisel3] object definitiveMacro {

  def impl(c: whitebox.Context)(annottees: c.Expr[Any]*): c.Expr[Any] = {
    import c.universe._
    val result = {
      val (deff, objOpt) = annottees.map(_.tree).toList match {
        case Seq(c, o) => (c, Some(o))
        case Seq(c)    => (c, None)
      }
      deff match {
        case aDef: DefDef => 
          val name = aDef.name
          val clzname = TypeName(name + "___MACRO_GENERATED")
          val args = aDef.vparamss
          val firstArgs = args.dropRight(1).map(x => x.map(y => q"${y.name}: ${y.tpt}" ))
          val lastArg = args.last.head
          val input  = (lastArg.name, lastArg.tpt)
          q"""
            $aDef
            case class $clzname(...$firstArgs) extends chisel3.experimental.hierarchy.core.CustomParameterFunction[${input._2}, ${aDef.tpt}] {
              val args = Nil
              type I = ${input._2}
              type O = ${aDef.tpt}
              override def apply(${input._1}: I): O = ${aDef.rhs}
            }
            def $name(...$firstArgs): $clzname = ${clzname.toTermName}(...$firstArgs)
          """

        case other        => other
      }
      //deff
    }
    c.Expr[Any](result)
  }
}

private[chisel3] class definitive extends StaticAnnotation {
  def macroTransform(annottees: Any*): Any = macro definitiveMacro.impl
}

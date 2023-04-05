// SPDX-License-Identifier: Apache-2.0

package chisel3.naming

import scala.language.experimental.macros
import scala.annotation.StaticAnnotation
import scala.reflect.macros.blackbox

private[chisel3] object identifyMacro {

  def impl(c: blackbox.Context)(annottees: c.Expr[Any]*): c.Expr[Any] = {
    import c.universe._
    val result = {
      val (clz, objOpt) = annottees.map(_.tree).toList match {
        case Seq(c, o) => (c, Some(o))
        case Seq(c)    => (c, None)
        case _ =>
          throw new Exception(
            s"Internal Error: Please file an issue at https://github.com/chipsalliance/chisel3/issues: Match error: annottees.map(_.tree).toList=${annottees.map(_.tree).toList}"
          )
      }
      val newClz = clz match {
        case q"$mods trait $tpname[..$tparams] extends { ..$earlydefns } with ..$parents { $self => ..$stats }" =>
          val defname = TypeName(tpname.toString + c.freshName())
          val instname = TypeName(tpname.toString + c.freshName())
          stats.foreach { stat =>
            stat match {
              case aDef: DefDef if aDef.name.toString == "_traitModuleDefinitionIdentifierProposal" =>
                c.error(aDef.pos, s"Custom implementations of _traitModuleDefinitionIdentifierProposal are not allowed")
              case _ =>
            }
          }
          val newMethod = q"override protected def _traitModuleDefinitionIdentifierProposal = Some(${tpname.toString})"
          val newStats = newMethod +: stats
          (
            q"$mods trait $tpname[..$tparams] extends { ..$earlydefns } with ..$parents { $self => ..$newStats }",
          )
        case _ =>
          c.error(c.enclosingPosition, "Can only use @identify on traits, not classes, objects, vals, or defs")
          clz
      }
      objOpt match {
        case None => newClz
        case Some(o) =>
          q"""
            $newClz

            $o
          """
      }
    }
    c.Expr[Any](result)
  }
}

private[chisel3] class fixTraitIdentifier extends StaticAnnotation {
  def macroTransform(annottees: Any*): Any = macro identifyMacro.impl
}

// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

import scala.language.experimental.macros
import scala.annotation.StaticAnnotation
import scala.reflect.macros.whitebox

private[chisel3] object instantiableMacro {

  def impl(c: whitebox.Context)(annottees: c.Expr[Any]*): c.Expr[Any] = {
    import c.universe._
    def processBody(stats: Seq[Tree]): (Seq[Tree], Iterable[Tree]) = {
      val hierarchyExtensions = scala.collection.mutable.ArrayBuffer.empty[Tree]
      hierarchyExtensions += q"implicit val mg = new chisel3.internal.MacroGenerated{}"
      // Note the triple `_` prefixing `module` is to avoid conflicts if a user marks a 'val module'
      //  with @public; in this case, the lookup code is ambiguous between the generated `def module`
      //  function and the argument to the generated implicit class.
      val resultStats = stats.flatMap { stat =>
        stat match {
          case hasPublic: ValOrDefDef if hasPublic.mods.annotations.toString.contains("new public()") =>
            hasPublic match {
              case aDef: DefDef =>
                c.error(aDef.pos, s"Cannot mark a def as @public")
                Nil
              // For now, we only omit protected/private vals
              case aVal: ValDef
                  if aVal.mods.hasFlag(c.universe.Flag.PRIVATE) || aVal.mods.hasFlag(c.universe.Flag.PROTECTED) =>
                c.error(aVal.pos, s"Cannot mark a private or protected val as @public")
                Nil
              case aVal: ValDef =>
                hierarchyExtensions += atPos(aVal.pos)(q"def ${aVal.name} = ___module._lookup(_.${aVal.name})")
                if (aVal.name.toString == aVal.children.last.toString) Nil else Seq(aVal)
              case other => Seq(other)
            }
          case other => Seq(other)
        }
      }
      (resultStats, hierarchyExtensions)
    }
    val result = {
      val (clz, objOpt) = annottees.map(_.tree).toList match {
        case Seq(c, o) => (c, Some(o))
        case Seq(c)    => (c, None)
      }
      val (newClz, implicitClzs, tpname) = clz match {
        case q"$mods class $tpname[..$tparams] $ctorMods(...$paramss) extends { ..$earlydefns } with ..$parents { $self => ..$stats }" =>
          val defname = TypeName(tpname + c.freshName())
          val instname = TypeName(tpname + c.freshName())
          val lensename = TypeName(tpname + c.freshName())
          val (newStats, hierarchyExtensions) = processBody(stats)
          val argTParams = tparams.map(_.name)
          (
            q""" $mods class $tpname[..$tparams] $ctorMods(...$paramss) extends { ..$earlydefns } with ..$parents { $self => ..$newStats } """,
            Seq(
              q"""implicit class $defname[..$tparams](___module: chisel3.experimental.hierarchy.core.Definition[$tpname[..$argTParams]]) { ..$hierarchyExtensions }""",
              q"""implicit class $instname[..$tparams](___module: chisel3.experimental.hierarchy.core.Instance[$tpname[..$argTParams]]) { ..$hierarchyExtensions } """,
              q"""implicit class $lensename[..$tparams](___module: chisel3.experimental.hierarchy.core.Lense[$tpname[..$argTParams]]) { ..$hierarchyExtensions } """
            ),
            tpname
          )
        case q"$mods trait $tpname[..$tparams] extends { ..$earlydefns } with ..$parents { $self => ..$stats }" =>
          val defname = TypeName(tpname + c.freshName())
          val instname = TypeName(tpname + c.freshName())
          val lensename = TypeName(tpname + c.freshName())
          val (newStats, hierarchyExtensions) = processBody(stats)
          val argTParams = tparams.map(_.name)
          (
            q"$mods trait $tpname[..$tparams] extends { ..$earlydefns } with ..$parents { $self => ..$newStats }",
            Seq(
              q"""implicit class $defname[..$tparams](___module: chisel3.experimental.hierarchy.core.Definition[$tpname[..$argTParams]]) { ..$hierarchyExtensions }""",
              q"""implicit class $instname[..$tparams](___module: chisel3.experimental.hierarchy.core.Instance[$tpname[..$argTParams]]) { ..$hierarchyExtensions } """,
              q"""implicit class $lensename[..$tparams](___module: chisel3.experimental.hierarchy.core.Lense[$tpname[..$argTParams]]) { ..$hierarchyExtensions } """
            ),
            tpname
          )
      }
      val newObj = objOpt match {
        case None => q"""object ${tpname.toTermName} { ..$implicitClzs } """
        case Some(obj @ q"$mods object $tname extends { ..$earlydefns } with ..$parents { $self => ..$body }") =>
          q"""
            $mods object $tname extends { ..$earlydefns } with ..$parents { $self =>
              ..$implicitClzs
              ..$body
            }
          """
      }
      q"""
        $newClz

        $newObj
      """
    }
    c.Expr[Any](result)
  }
}

private[chisel3] class instantiable extends StaticAnnotation {
  def macroTransform(annottees: Any*): Any = macro instantiableMacro.impl
}
private[chisel3] class public extends StaticAnnotation

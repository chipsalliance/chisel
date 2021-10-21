// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

import scala.language.experimental.macros
import scala.annotation.StaticAnnotation
import scala.reflect.macros.whitebox


private[chisel3] object instantiableMacro {

  def impl(c: whitebox.Context)(annottees: c.Expr[Any]*): c.Expr[Any] = {
    import c.universe._
    def processBody(stats: Seq[Tree]): (Seq[Tree], Iterable[Tree]) = {
      val extensions = scala.collection.mutable.ArrayBuffer.empty[Tree]
      extensions += q"implicit val mg = new chisel3.internal.MacroGenerated{}"
      val resultStats = stats.flatMap {
        case x @ q"@public val $tpname: $tpe = $name" if tpname.toString() == name.toString() =>
          extensions += atPos(x.pos)(q"def $tpname = module._lookup(_.$tpname)")
          Nil
        case x @ q"@public val $tpname: $tpe = $_" =>
          extensions += atPos(x.pos)(q"def $tpname = module._lookup(_.$tpname)")
          Seq(x)
        case x @ q"@public val $tpname: $tpe" =>
          extensions += atPos(x.pos)(q"def $tpname = module._lookup(_.$tpname)")
          Seq(x)
        case x @ q"@public lazy val $tpname: $tpe = $_" =>
          extensions += atPos(x.pos)(q"def $tpname = module._lookup(_.$tpname)")
          Seq(x)
        case other =>
          Seq(other)
      }
      (resultStats, extensions)
    }
    val result = {
      val (clz, objOpt) = annottees.map(_.tree).toList match {
        case Seq(c, o) => (c, Some(o))
        case Seq(c) => (c, None)
      }
      val (newClz, implicitClzs, tpname) = clz match {
        case q"$mods class $tpname[..$tparams] $ctorMods(...$paramss) extends { ..$earlydefns } with ..$parents { $self => ..$stats }" =>
          val defname = TypeName(tpname + c.freshName())
          val instname = TypeName(tpname + c.freshName())
          val (newStats, extensions) = processBody(stats)
          (q""" $mods class $tpname[..$tparams] $ctorMods(...$paramss) extends { ..$earlydefns } with ..$parents with chisel3.experimental.hierarchy.IsInstantiable { $self => ..$newStats } """,
           Seq(q"""implicit class $defname(module: chisel3.experimental.hierarchy.Definition[$tpname]) { ..$extensions }""",
               q"""implicit class $instname(module: chisel3.experimental.hierarchy.Instance[$tpname]) { ..$extensions } """),
           tpname)
        case q"$mods trait $tpname[..$tparams] extends { ..$earlydefns } with ..$parents { $self => ..$stats }" =>
          val defname = TypeName(tpname + c.freshName())
          val instname = TypeName(tpname + c.freshName())
          val (newStats, extensions) = processBody(stats)
          (q"$mods trait $tpname[..$tparams] extends { ..$earlydefns } with ..$parents with chisel3.experimental.hierarchy.IsInstantiable { $self => ..$newStats }",
           Seq(q"""implicit class $defname(module: chisel3.experimental.hierarchy.Definition[$tpname]) { ..$extensions }""",
               q"""implicit class $instname(module: chisel3.experimental.hierarchy.Instance[$tpname]) { ..$extensions } """),
           tpname)
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

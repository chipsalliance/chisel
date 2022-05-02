// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

import scala.language.experimental.macros
import scala.annotation.StaticAnnotation
import scala.reflect.macros.whitebox

private[chisel3] object instantiableMacro {

  def impl(c: whitebox.Context)(annottees: c.Expr[Any]*): c.Expr[Any] = {
    import c.universe._
    def processBody(
      tpname:  TypeName,
      tparams: Iterable[Tree],
      stats:   Seq[Tree]
    ): (Seq[Tree], Iterable[Tree], Iterable[Tree]) = {
      val extensions = scala.collection.mutable.ArrayBuffer.empty[Tree]
      extensions += q"implicit val mg = new chisel3.internal.MacroGenerated{}"

      val nameMappings = scala.collection.mutable.ArrayBuffer.empty[Tree]

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
                extensions += atPos(aVal.pos)(q"def ${aVal.name} = ___module._lookup(_.${aVal.name})")
                val name = Literal(Constant(aVal.name.decodedName.toString))
                val wildcards = tparams.map(_ => "_")
                nameMappings += atPos(aVal.pos)(
                  q"___nameToFunc(${name}) = { x: Any => x.asInstanceOf[$tpname[..$wildcards]].${aVal.name} }"
                )
                if (aVal.name.toString == aVal.children.last.toString) Nil else Seq(aVal)
              case other => Seq(other)
            }
          case other => Seq(other)
        }
      }
      (resultStats, extensions, nameMappings)
    }
    val result = {
      val (clz, objOpt) = annottees.map(_.tree).toList match {
        case Seq(c, o) => (c, Some(o))
        case Seq(c)    => (c, None)
      }
      val (newClz, implicitClzs, tpname) = clz match {
        case q"$mods class $tpname[..$tparams] $ctorMods(...$paramss) extends { ..$earlydefns } with ..$parents { $self => ..$stats }" =>
          val defname = TypeName(tpname + c.freshName())
          val toFreezable = TermName(tpname + c.freshName())
          val freezableWrapper = TypeName(tpname + c.freshName())
          val argTParams = tparams.map(_.name)
          val (newStats, extensions, nameMappings) = processBody(tpname, tparams, stats)

          (
            q""" $mods class $tpname[..$tparams] $ctorMods(...$paramss) extends { ..$earlydefns } with ..$parents { $self => ..$newStats } """,
            Seq(
              q"""implicit class $defname[..$tparams](___module: chisel3.experimental.hierarchy.core.Wrapper[$tpname[..$argTParams]]) { ..$extensions }""",
              q"""implicit class $freezableWrapper[..$tparams](___module: $tpname[..$argTParams]) {
                def toFreezable = $toFreezable[..$argTParams].toUnderlying(___module)
              }""",
              q"""implicit def $toFreezable[..$tparams] = new chisel3.experimental.hierarchy.core.ToFreezable[$tpname[..$argTParams]] { 
                val ___nameToFunc = scala.collection.mutable.HashMap.empty[String, Any => Any]
                ..$nameMappings
                override type X = $tpname[..$argTParams]
                def toUnderlying(x: X) = {
                  new chisel3.experimental.hierarchy.core.Freezable[X](x, ___nameToFunc.toMap)
                }
              }"""
            ),
            tpname
          )
        case q"$mods trait $tpname[..$tparams] extends { ..$earlydefns } with ..$parents { $self => ..$stats }" =>
          val defname = TypeName(tpname + c.freshName())
          val (newStats, extensions, nameMappings) = processBody(tpname, tparams, stats)
          val argTParams = tparams.map(_.name)
          (
            q"$mods trait $tpname[..$tparams] extends { ..$earlydefns } with ..$parents { $self => ..$newStats }",
            Seq(
              q"""implicit class $defname[..$tparams](___module: chisel3.experimental.hierarchy.core.Wrapper[$tpname[..$argTParams]]) { ..$extensions }"""
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

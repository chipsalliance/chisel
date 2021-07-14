package chisel3.internal

import scala.language.experimental.macros
import scala.annotation.{StaticAnnotation, compileTimeOnly}
import scala.reflect.macros.whitebox
import scala.reflect.macros.blackbox


object InstanceHandleMacro {
  def impl(c: blackbox.Context)(annottees: c.Expr[Any]*) = {
    import c.universe._
    val inputs = annottees.map(_.tree).toList
    val (annottee, expandees) = inputs match {
      case (param: ValDef) :: (rest @ (_ :: _)) => (param, rest)
      case (param: TypeDef) :: (rest @ (_ :: _)) => (param, rest)
      case _ => (EmptyTree, inputs)
    }
    //println((annottee, expandees))
    val outputs = expandees
    c.Expr[Any](Block(outputs, Literal(Constant(()))))
  }
}
object instanceMacro {
  def impl(c: whitebox.Context)(annottees: c.Expr[Any]*): c.Expr[Any] = {
    import c.universe._
    import Flag._
    val result = {
      //println(annottees)
      annottees.map(_.tree).toList match {
        case Seq(clz@q"$mods class $tpname[..$tparams] $ctorMods(...$paramss) extends { ..$earlydefns } with ..$parents { $self => ..$stats }") =>
          val name2 = TypeName(tpname.+(c.freshName()))
          val extensions = scala.collection.mutable.ArrayBuffer[c.universe.Tree]()
          val newBodies = stats.flatMap {
            case q"@public val $tpname: $tpe = $name" if tpname.toString() == name.toString() =>
              extensions += q"def $tpname: $tpe = module(_.$tpname)"
              Nil
            case x@q"@public val $tpname: $tpe = $_" =>
              extensions += q"def $tpname: $tpe = module(_.$tpname)"
              Seq(x)
            case other => Seq(other)
          }
          q"""
            $mods class $tpname[..$tparams] $ctorMods(...$paramss) extends { ..$earlydefns } with ..$parents { $self => ..${newBodies.toSeq} }

            object ${tpname.toTermName} {
              implicit class $name2(module: chisel3.Instance[$tpname]) { ..$extensions }
            }
          """
        case Seq(clz@q"$_ class $tpname[..$_] $_(...$_) extends { ..$_ } with ..$_ { $_ => ..$stats }",
                 obj@q"$mods object $tname extends { ..$earlydefns } with ..$parents { $self => ..$body }") =>
          val name2 = TypeName(tpname.+(c.freshName()))
          val bodies = stats.collect {
            case q"@public val $tpname: $tpe = $other" => q"def $tpname: $tpe = module(_.$tpname)"
          }
          val implicitClz = q"""implicit class $name2(module: chisel3.Instance[$tpname]) { ..$bodies }"""
          q"""
            $clz

            $mods object $tname extends { ..$earlydefns } with ..$parents { $self =>
              $implicitClz

              ..$body
            }
          """
      }
    }
    c.Expr[Any](result)
  }
}

class instance extends StaticAnnotation {
  def macroTransform(annottees: Any*): Any = macro instanceMacro.impl
}
class public extends StaticAnnotation {
  def macroTransform(annottees: Any*): Any = macro InstanceHandleMacro.impl
}
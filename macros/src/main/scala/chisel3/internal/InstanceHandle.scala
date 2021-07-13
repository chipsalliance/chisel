package chisel3.internal

import scala.language.experimental.macros
import scala.annotation.{StaticAnnotation, compileTimeOnly}
import scala.reflect.macros.whitebox
import scala.reflect.macros.blackbox


class InstanceHandleMacro {
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
object helloMacro {
  def impl(c: blackbox.Context)(annottees: c.Expr[Any]*): c.Expr[Any] = {
    import c.universe._
    import Flag._
    val result = {
      annottees.map(_.tree).toList match {
        case q"$mods class $tpname[..$tparams] $ctorMods(...$paramss) extends { ..$earlydefns } with ..$parents { $self => ..$stats }" :: Nil =>
          val name2 = TypeName(tpname.+(c.freshName()))
          val bodies = stats.collect {
            case q"@public val $tpname: $tpe = $other" => q"def $tpname: $tpe = module(_.$tpname)"
          }
          q"""
            $mods class $tpname[..$tparams] $ctorMods(...$paramss) extends { ..$earlydefns } with ..$parents { $self => ..$stats }
            implicit class $name2(module: Instance[$tpname]) {
              ..$bodies
            }
          """
        case List(x) => println(x); x
      }
    }
    c.Expr[Any](result)
  }
}

class interface extends StaticAnnotation {
  def macroTransform(annottees: Any*): Any = macro helloMacro.impl
}
class public extends StaticAnnotation
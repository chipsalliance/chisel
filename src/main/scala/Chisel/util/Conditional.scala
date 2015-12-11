// See LICENSE for license details.

/** Conditional blocks.
  */

package Chisel

import scala.language.reflectiveCalls
import scala.language.experimental.macros
import scala.reflect.runtime.universe._
import scala.reflect.macros.blackbox._

/** This is identical to [[Chisel.when when]] with the condition inverted */
object unless {  // scalastyle:ignore object.name
  def apply(c: Bool)(block: => Unit) {
    when (!c) { block }
  }
}

class SwitchContext[T <: Bits](cond: T) {
  def is(v: Iterable[T])(block: => Unit) {
    if (!v.isEmpty) when (v.map(_.asUInt === cond.asUInt).reduce(_||_)) { block }
  }
  def is(v: T)(block: => Unit) { is(Seq(v))(block) }
  def is(v: T, vr: T*)(block: => Unit) { is(v :: vr.toList)(block) }
}

/** An object for separate cases in [[Chisel.switch switch]]
  * It is equivalent to a [[Chisel.when$ when]] block comparing to the condition
  * Use outside of a switch statement is illegal */
object is {   // scalastyle:ignore object.name
  // Begin deprecation of non-type-parameterized is statements.
  def apply(v: Iterable[Bits])(block: => Unit) {
    require(false, "The 'is' keyword may not be used outside of a switch.")
  }

  def apply(v: Bits)(block: => Unit) {
    require(false, "The 'is' keyword may not be used outside of a switch.")
  }

  def apply(v: Bits, vr: Bits*)(block: => Unit) {
    require(false, "The 'is' keyword may not be used outside of a switch.")
  }
}

/** Conditional logic to form a switch block
  * @example
  * {{{ ... // default values here
  * switch ( myState ) {
  *   is( state1 ) {
  *     ... // some logic here
  *   }
  *   is( state2 ) {
  *     ... // some logic here
  *   }
  * } }}}*/
object switch {  // scalastyle:ignore object.name
  def apply[T <: Bits](cond: T)(x: => Unit): Unit = macro impl
  def impl(c: Context)(cond: c.Tree)(x: c.Tree): c.Tree = { import c.universe._
    val sc = c.universe.internal.reificationSupport.freshTermName("sc")
    def extractIsStatement(tree: Tree): List[c.universe.Tree] = tree match {
      case q"Chisel.is.apply( ..$params )( ..$body )" => List(q"$sc.is( ..$params )( ..$body )")
      case b => throw new Exception(s"Cannot include blocks that do not begin with is() in switch.")
    }
    val q"..$body" = x
    val ises = body.flatMap(extractIsStatement(_))
    q"""{ val $sc = new SwitchContext($cond); ..$ises }"""
  }
}

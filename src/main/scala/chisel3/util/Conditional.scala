// See LICENSE for license details.

/** Conditional blocks.
  */

package chisel3.util

import scala.language.reflectiveCalls
import scala.language.experimental.macros
import scala.reflect.runtime.universe._
import scala.reflect.macros.blackbox._

import chisel3._

object unless {  // scalastyle:ignore object.name
  /** Does the same thing as [[when$ when]], but with the condition inverted.
    */
  def apply(c: Bool)(block: => Unit) {
    when (!c) { block }
  }
}

/** Implementation details for [[switch]]. See [[switch]] and [[chisel3.util.is is]] for the
  * user-facing API.
  * @note DO NOT USE. This API is subject to change without warning.
  */
class SwitchContext[T <: Bits](cond: T, whenContext: Option[WhenContext], lits: Set[BigInt]) {
  def is(v: Iterable[T])(block: => Unit): SwitchContext[T] = {
    if (!v.isEmpty) {
      val newLits = v.map { w =>
        require(w.isLit, "is conditions must be literals!")
        val value = w.litValue
        require(!lits.contains(value), "all is conditions must be mutually exclusive!")
        value
      }
      // def instead of val so that logic ends up in legal place
      def p = v.map(_.asUInt === cond.asUInt).reduce(_||_)
      whenContext match {
        case Some(w) => new SwitchContext(cond, Some(w.elsewhen(p)(block)), lits ++ newLits)
        case None => new SwitchContext(cond, Some(when(p)(block)), lits ++ newLits)
      }
    } else {
      this
    }
  }
  def is(v: T)(block: => Unit): SwitchContext[T] = is(Seq(v))(block)
  def is(v: T, vr: T*)(block: => Unit): SwitchContext[T] = is(v :: vr.toList)(block)
}

/** Use to specify cases in a [[switch]] block, equivalent to a [[when$ when]] block comparing to
  * the condition variable.
  *
  * @note illegal outside a [[switch]] block
  * @note must be a literal
  * @note each is must be mutually exclusive
  * @note dummy implementation, a macro inside [[switch]] transforms this into the actual
  * implementation
  */
object is {   // scalastyle:ignore object.name
  // TODO: Begin deprecation of non-type-parameterized is statements.
  /** Executes `block` if the switch condition is equal to any of the values in `v`.
    */
  def apply(v: Iterable[Bits])(block: => Unit) {
    require(false, "The 'is' keyword may not be used outside of a switch.")
  }

  /** Executes `block` if the switch condition is equal to `v`.
    */
  def apply(v: Bits)(block: => Unit) {
    require(false, "The 'is' keyword may not be used outside of a switch.")
  }

  /** Executes `block` if the switch condition is equal to any of the values in the argument list.
    */
  def apply(v: Bits, vr: Bits*)(block: => Unit) {
    require(false, "The 'is' keyword may not be used outside of a switch.")
  }
}

/** Conditional logic to form a switch block. See [[is$ is]] for the case API.
  *
  * @example {{{
  * switch (myState) {
  *   is (state1) {
  *     // some logic here that runs when myState === state1
  *   }
  *   is (state2) {
  *     // some logic here that runs when myState === state2
  *   }
  * }
  * }}}
  */
object switch {  // scalastyle:ignore object.name
  def apply[T <: Bits](cond: T)(x: => Unit): Unit = macro impl
  def impl(c: Context)(cond: c.Tree)(x: c.Tree): c.Tree = { import c.universe._
    val q"..$body" = x
    val res = body.foldLeft(q"""new SwitchContext($cond, None, Set.empty)""") {
      case (acc, tree) => tree match {
        // TODO: remove when Chisel compatibility package is removed
        case q"Chisel.`package`.is.apply( ..$params )( ..$body )" => q"$acc.is( ..$params )( ..$body )"
        case q"chisel3.util.is.apply( ..$params )( ..$body )" => q"$acc.is( ..$params )( ..$body )"
        case b => throw new Exception(s"Cannot include blocks that do not begin with is() in switch.")
      }
    }
    q"""{ $res }"""
  }
}

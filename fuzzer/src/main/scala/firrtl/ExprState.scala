package firrtl.fuzzer

import firrtl.ir.{Expression, Reference, Type}

import scala.language.higherKinds

/** A typeclass for types that represent the state of a random expression generator
  */
trait ExprState[State] {

  /** Creates a [[StateGen]] that adds a reference to the input state and returns that reference
    */
  def withRef[Gen[_]: GenMonad](ref: Reference): StateGen[State, Gen, Reference]

  /** Creates a [[StateGen]] that returns an [[firrtl.ir.Expression Expression]] with the specified type
    */
  def exprGen[Gen[_]: GenMonad](tpe: Type): StateGen[State, Gen, Expression]

  /** Gets the set of unbound references
    */
  def unboundRefs(s: State): Set[Reference]

  /** Gets the maximum allowed width of any Expression
    */
  def maxWidth(s: State): Int
}

object ExprState {
  def apply[S: ExprState] = implicitly[ExprState[S]]
}

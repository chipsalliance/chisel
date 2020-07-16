package firrtl.fuzzer

import scala.language.higherKinds

/** Wraps a function that takes a function an produces a random state transition and value
  *
  * @tparam State the type of the initial and resulting state of this random computation
  * @tparam Gen the random context that wraps the return value of this function
  * @tparam A the type of the value returned by this function
  */
final case class StateGen[State, Gen[_], A](run: State => Gen[(State, A)]) {

  /** Creates a new [[StateGen]] that applies the function to the result of this [[StateGen]] and flattens the result
    */
  def flatMap[B](fn: A => StateGen[State, Gen, B])(implicit GM: GenMonad[Gen]): StateGen[State, Gen, B] = {
    StateGen { state =>
      GM.flatMap(run(state)) { case (sx, a) =>
        fn(a).run(sx)
      }
    }
  }

  /** Creates a new [[StateGen]] that applies the function to the result of this [[StateGen]]
    */
  def map[B](f: A => B)(implicit GM: GenMonad[Gen]): StateGen[State, Gen, B] = StateGen { state =>
    GM.map(run(state)) { case (sx, a) =>
      sx -> f(a)
    }
  }

  /** Returns the same [[StateGen]] but with a wider result type parameter
    */
  def widen[B >: A](implicit GM: GenMonad[Gen]): StateGen[State, Gen, B] = StateGen { state =>
    GM.map(run(state)) { case (state, a) => state -> a }
  }
}

object StateGen {

  /** Takes a constant value and turns it into a [[StateGen]]
    */
  def pure[S, Gen[_]: GenMonad, A](a: A): StateGen[S, Gen, A] = {
    StateGen((s: S) => GenMonad[Gen].const(s -> a))
  }

  /** Takes a random value generator and turns it into a [[StateGen]]
    */
  def liftG[S, Gen[_]: GenMonad, A](ga: Gen[A]): StateGen[S, Gen, A] = {
    StateGen((s: S) => GenMonad[Gen].map(ga)(s -> _))
  }

  /** Creates a [[StateGen]] produces a value from the input state without modifying it
    */
  def inspect[S, Gen[_]: GenMonad, A](fn: S => A): StateGen[S, Gen, A] = {
    StateGen(s => GenMonad[Gen].const((s, fn(s))))
  }

  /** Creates a [[StateGen]] produces a random value from the input state without modifying it
    */
  def inspectG[S, Gen[_]: GenMonad, A](fn: S => Gen[A]): StateGen[S, Gen, A] = {
    StateGen(s => GenMonad[Gen].map(fn(s)) { s -> _ })
  }
}

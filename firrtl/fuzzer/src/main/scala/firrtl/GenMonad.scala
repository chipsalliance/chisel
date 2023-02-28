// SPDX-License-Identifier: Apache-2.0

package firrtl.fuzzer

import scala.language.higherKinds

/** Monads that represent a random value generator
  */
trait GenMonad[Gen[_]] {

  /** Creates a new generator that applies the function to the output of the first generator and flattens the result
    */
  def flatMap[A, B](a: Gen[A])(f: A => Gen[B]): Gen[B]

  /** Creates a new generator that applies the function to the output of the first generator
    */
  def map[A, B](a: Gen[A])(f: A => B): Gen[B]

  /** Flattens a nested generator into a single generator
    */
  def flatten[A](gga: Gen[Gen[A]]): Gen[A] = flatMap(gga)(ga => ga)

  /** Creates a generator that produces values uniformly distributed across the range
    *
    * The generated values are inclusive of both min and max.
    */
  def choose(min: Int, max: Int): Gen[Int]

  /** Creates a generator that uniformly selects from a list of items
    */
  def oneOf[A](items: A*): Gen[A]

  /** Creates a generator that always returns the same value
    */
  def const[A](c: A): Gen[A]

  /** Returns the same generator but with a wider type parameter
    */
  def widen[A, B >: A](ga: Gen[A]): Gen[B]

  /** Runs the given generator and returns the generated value
    */
  def generate[A](ga: Gen[A]): A
}

object GenMonad {
  def apply[Gen[_]: GenMonad] = implicitly[GenMonad[Gen]]

  /** Creates a generator that pick between true and false
    */
  def bool[Gen[_]: GenMonad]: Gen[Boolean] = GenMonad[Gen].oneOf(true, false)

  /** Creates a generator that generates values based on the weights paired with each value
    */
  def frequency[Gen[_]: GenMonad, A](pairs: (Int, A)*): Gen[A] = {
    assert(pairs.forall(_._1 > 0))
    assert(pairs.size >= 1)
    val total = pairs.map(_._1).sum
    GenMonad[Gen].map(GenMonad[Gen].choose(1, total)) { startnum =>
      var idx = 0
      var num = startnum - pairs(idx)._1
      while (num > 0) {
        idx += 1
        num -= pairs(idx)._1
      }
      pairs(idx)._2
    }
  }

  /** Provides extension methods like .flatMap and .flatten for [[GenMonad]]s
    */
  object syntax {
    final class GenMonadOps[Gen[_], A](ga: Gen[A])(implicit GM: GenMonad[Gen]) {
      def flatMap[B](f: A => Gen[B]): Gen[B] = {
        GM.flatMap(ga)(f)
      }
      def map[B](f: A => B): Gen[B] = {
        GM.map(ga)(f)
      }
      def widen[B >: A]: Gen[B] = {
        GM.widen[A, B](ga)
      }
      def generate(): A = {
        GM.generate(ga)
      }
    }

    final class GenMonadFlattenOps[Gen[_], A](gga: Gen[Gen[A]])(implicit GM: GenMonad[Gen]) {
      def flatten: Gen[A] = GM.flatten(gga)
    }

    implicit def genMonadOps[Gen[_]: GenMonad, A](ga: Gen[A]): GenMonadOps[Gen, A] =
      new GenMonadOps(ga)

    implicit def genMonadFlattenOps[Gen[_]: GenMonad, A](gga: Gen[Gen[A]]): GenMonadFlattenOps[Gen, A] =
      new GenMonadFlattenOps(gga)
  }
}

// SPDX-License-Identifier: Apache-2.0

package circt

/** A collection of implicit classes to provide additional methods to existing types */
object Implicits {

  /** Helpers for working with Boolean */
  implicit class BooleanImplicits(a: Boolean) {

    /** Construct an Option from a Boolean. */
    def option[A](b: => A): Option[A] =
      if (a)
        Some(b)
      else
        None
  }

}

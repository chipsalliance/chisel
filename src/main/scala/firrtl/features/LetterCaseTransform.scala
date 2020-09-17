// SPDX-License-Identifier: Apache-2.0

package firrtl.features

import firrtl.Namespace
import firrtl.transforms.ManipulateNames

import scala.reflect.ClassTag

/** Parent of transforms that do change the letter case of names in a FIRRTL circuit */
abstract class LetterCaseTransform[A <: ManipulateNames[_]: ClassTag] extends ManipulateNames[A] {

  protected def newName: String => String

  final def manipulate = (a: String, ns: Namespace) =>
    newName(a) match {
      case `a` => None
      case b   => Some(ns.newName(b))
    }
}

/** Convert all FIRRTL names to lowercase */
final class LowerCaseNames extends LetterCaseTransform[LowerCaseNames] {
  override protected def newName = (a: String) => a.toLowerCase
}

/** Convert all FIRRTL names to UPPERCASE */
final class UpperCaseNames extends LetterCaseTransform[UpperCaseNames] {
  override protected def newName = (a: String) => a.toUpperCase
}

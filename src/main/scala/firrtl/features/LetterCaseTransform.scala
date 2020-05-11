// See LICENSE for license details.

package firrtl.features

import firrtl.{analyses, Namespace, passes, Transform}
import firrtl.options.Dependency
import firrtl.stage.Forms
import firrtl.transforms.ManipulateNames

/** Parent of transforms that do change the letter case of names in a FIRRTL circuit */
abstract class LetterCaseTransform extends ManipulateNames {
  override def prerequisites = Seq(Dependency(passes.LowerTypes))
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Forms.LowEmitters
  override def invalidates(a: Transform) = a match {
    case _: analyses.GetNamespace => true
    case _                        => false
  }

  protected def newName: String => String

  final def condition = _ => true

  final def manipulate = (a: String, ns: Namespace) => newName(a) match {
    case `a` => a
    case b   => ns.newName(b)
  }
}

/** Convert all FIRRTL names to lowercase */
final class LowerCaseNames extends LetterCaseTransform {
  override protected def newName = (a: String) => a.toLowerCase
}

/** Convert all FIRRTL names to UPPERCASE */
final class UpperCaseNames extends LetterCaseTransform {
  override protected def newName = (a: String) => a.toUpperCase
}

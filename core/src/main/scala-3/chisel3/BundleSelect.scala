// SPDX-License-Identifier: Apache-2.0

package chisel3

/**
  * Unlike in Scala 2, in Scala 3 the compiler no longer infers the
  * refined structural type for an anonymous subclass - it infers the
  * unrefined type for anonymous Bundles, which will always be Bundle.
  *
  * This extension method allows accessing elements of an anonymous
  * Bundle even if its refined type is not inferred - without it the
  * compiler will try to grab the element from class Bundle itself and
  * will obviously error.
  */
extension (b: Bundle) {
  def selectDynamic(field: String): Any = b.elements(field)
}

// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.experimental.{AutoCloneType, requireIsChiselType, DataMirror}
import scala.collection.immutable.ListMap

// An example of how Record might be extended
// In this case, CustomBundle is a Record constructed from a Tuple of (String, Data)
//   it is a possible implementation of a programmatic "Bundle"
//   (and can by connected to MyBundle below)
final class CustomBundle(elts: (String, Data)*) extends Record with AutoCloneType {
  val elements = ListMap(elts.map {
    case (field, elt) =>
      requireIsChiselType(elt)
      field -> elt
  }: _*)
  def apply(elt: String): Data = elements(elt)
}

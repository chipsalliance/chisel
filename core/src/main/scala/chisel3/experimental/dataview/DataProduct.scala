// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.dataview

import chisel3.{Data, getRecursiveFields}

import scala.annotation.implicitNotFound

/** Typeclass interface for getting elements of type [[Data]]
  *
  * This is needed for validating [[DataView]]s targeting type `A`.
  * Can be thought of as "can be the Target of a DataView".
  * @tparam A Type that has elements of type [[Data]]
  */
@implicitNotFound("Could not find implicit value for DataProduct[${A}].\nPlease see <docs link>")
trait DataProduct[-A] {
  /** Provides [[Data]] elements within some parent type
    *
    * @param a Parent type
    * @param path Hierarchical path to current signal (for error reporting)
    * @return Data elements and associated String paths (Strings for error reporting only!)
    */
  def dataIterator(a: A, path: String): Iterator[(Data, String)]
}

object DataProduct {
  implicit val dataDataProduct: DataProduct[Data] = new DataProduct[Data] {
    def dataIterator(a: Data, path: String): Iterator[(Data, String)] =
      getRecursiveFields(a, path).iterator
  }
}
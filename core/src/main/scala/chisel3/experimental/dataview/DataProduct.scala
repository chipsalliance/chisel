// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.dataview

import chisel3.experimental.BaseModule
import chisel3.{Data, getRecursiveFields}

import scala.annotation.implicitNotFound

/** Typeclass interface for getting elements of type [[Data]]
  *
  * This is needed for validating [[DataView]]s targeting type `A`.
  * Can be thought of as "can be the Target of a DataView".
  *
  * Chisel provides some implementations in [[DataProduct$ object DataProduct]] that are available
  * by default in the implicit scope.
  *
  * @tparam A Type that has elements of type [[Data]]
  * @see [[https://www.chisel-lang.org/chisel3/docs/explanations/dataview#dataproduct Detailed Documentation]]
  */
@implicitNotFound("Could not find implicit value for DataProduct[${A}].\n" +
  "Please see https://www.chisel-lang.org/chisel3/docs/explanations/dataview#dataproduct")
trait DataProduct[-A] {
  /** Provides [[Data]] elements within some containing object
    *
    * @param a Containing object
    * @param path Hierarchical path to current signal (for error reporting)
    * @return Data elements and associated String paths (Strings for error reporting only!)
    */
  def dataIterator(a: A, path: String): Iterator[(Data, String)]

  /** Returns a checker to test if the containing object contains a `Data` object
    * @note Implementers may want to override if iterating on all `Data` is expensive for `A` and `A`
    *       will primarily be used in `PartialDataViews`
    * @note The returned value is a function, not a true Set, but is describing the functionality of
    *       Set containment
    * @param a Containing object
    * @return A checker that itself returns True if a given `Data` is contained in `a`
    *         as determined by an `==` test
    */
  def dataSet(a: A): Data => Boolean = dataIterator(a, "").map(_._1).toSet
}

/** Encapsulating object for automatically provided implementations of [[DataProduct]]
  *
  * @note DataProduct implementations provided in this object are available in the implicit scope
  */
object DataProduct {
  /** [[DataProduct]] implementation for [[Data]] */
  implicit val dataDataProduct: DataProduct[Data] = new DataProduct[Data] {
    def dataIterator(a: Data, path: String): Iterator[(Data, String)] =
      getRecursiveFields.lazily(a, path).iterator
  }

  /** [[DataProduct]] implementation for [[BaseModule]] */
  implicit val userModuleDataProduct: DataProduct[BaseModule] = new DataProduct[BaseModule] {
    def dataIterator(a: BaseModule, path: String): Iterator[(Data, String)] = {
      a.getIds.iterator.flatMap {
        case d: Data if d.getOptionRef.isDefined => // Using ref to decide if it's truly hardware in the module
          Seq(d -> s"${path}.${d.instanceName}")
        case b: BaseModule => dataIterator(b, s"$path.${b.instanceName}")
        case _ => Seq.empty
      }
    }
    // Overridden for performance
    override def dataSet(a: BaseModule): Data => Boolean = {
      val lastId = a._lastId // Not cheap to compute
      // Return a function
      e => e._id > a._id && e._id <= lastId
    }
  }
}

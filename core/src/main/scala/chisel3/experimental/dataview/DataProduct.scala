// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.dataview

import chisel3.experimental.BaseModule
import chisel3.{Data, Vec, getRecursiveFields}

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

/** Low priority built-in implementations of [[DataProduct]]
  *
  * @note This trait exists so that `dataDataProduct` can be lower priority than `seqDataProduct` to resolve ambiguity
  */
sealed trait LowPriorityDataProduct {

  /** [[DataProduct]] implementation for [[Data]] */
  implicit val dataDataProduct: DataProduct[Data] = new DataProduct[Data] {
    def dataIterator(a: Data, path: String): Iterator[(Data, String)] =
      getRecursiveFields.lazily(a, path).iterator
  }
}

/** Encapsulating object for built-in implementations of [[DataProduct]]
  *
  * @note DataProduct implementations provided in this object are available in the implicit scope
  */
object DataProduct extends LowPriorityDataProduct {
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

  /** [[DataProduct]] implementation for any `Seq[A]` where `A` has an implementation of `DataProduct`. */
  implicit def seqDataProduct[A : DataProduct]: DataProduct[Seq[A]] = new DataProduct[Seq[A]] {
    def dataIterator(a: Seq[A], path: String): Iterator[(Data, String)] = {
      val dpa = implicitly[DataProduct[A]]
      a.iterator
        .zipWithIndex
        .flatMap { case (elt, idx) =>
          dpa.dataIterator(elt, s"$path[$idx]")
        }
    }
  }

  /** [[DataProduct]] implementation for any [[Tuple2]] where each field has an implementation of `DataProduct`. */
  implicit def tuple2DataProduct[A : DataProduct, B : DataProduct]: DataProduct[(A, B)] = new DataProduct[(A, B)] {
    def dataIterator(tup: (A, B), path: String): Iterator[(Data, String)] = {
      val dpa = implicitly[DataProduct[A]]
      val dpb = implicitly[DataProduct[B]]
      val (a, b) = tup
      dpa.dataIterator(a, s"$path._1") ++ dpb.dataIterator(b, s"$path._2")
    }
  }
}

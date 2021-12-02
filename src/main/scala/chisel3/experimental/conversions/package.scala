
package chisel3.experimental

import chisel3._
import chisel3.experimental.DataMirror.internal.chiselTypeClone
import chisel3.experimental.dataview._
import scala.language.implicitConversions

/** Default implementations of [[DataProduct]] and [[DataView]] as well as implicit conversions to [[Data]] for some
  * Scala standard library types
  *
  * @note As this leans heavily on the experimental [[DataView]] feature, these APIs are experimental and subject to change
  */
package object conversions {

  // ****************************** Seq ******************************

  /** Provides implementations of [[DataProduct]] for any `Seq[A]` where `A` has an implementation of `DataProduct`. */
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

  /** Provides `DataView[Seq[A], Vec[B]]` for all `A` such that there exists `DataView[A, B]` */
  implicit def seqAsVecDataView[A : DataProduct, B <: Data](implicit dv: DataView[A, B]): DataView[Seq[A], Vec[B]] =
    // TODO this would need a better way to determine the prototype for the Vec
    DataView.mapping[Seq[A], Vec[B]](
      xs => Vec(xs.size, chiselTypeClone(xs.head.viewAs[B])), // xs.head is not correct in general
      { case (s, v) => s.zip(v).map { case (a, b) => a.viewAs[B] -> b } }
    )

  /** Implicit conversion between `Seq` and `Vec` */
  implicit def seq2Vec[A : DataProduct, B <: Data](xs: Seq[A])(implicit dv: DataView[A, B]): Vec[B] =
    xs.viewAs[Vec[B]]

  // ****************************** Tuple2 ******************************

  /** [[Data]] equivalent of Scala's [[Tuple2]]
    *
    * Users may not instantiate this class directly. Instead they should use the implicit conversion from `Tuple2`
    */
  final class HWTuple2[+A <: Data, +B <: Data] private[conversions] (val _1: A, val _2: B) extends Bundle

  /** Provides implementations of [[DataView]] for `(A, B)` to `HWTuple2[A, B]`  */
  implicit def view[T1 : DataProduct, T2 : DataProduct, V1 <: Data, V2 <: Data](
    implicit v1: DataView[T1, V1], v2: DataView[T2, V2]
  ): DataView[(T1, T2), HWTuple2[V1, V2]] =
    DataView.mapping(
      { case (a, b) => new HWTuple2(a.viewAs[V1].cloneType, b.viewAs[V2].cloneType)},
      { case ((a, b), hwt) =>
        Seq(a.viewAs[V1] -> hwt._1,
          b.viewAs[V2] -> hwt._2)
      }
    )

  /** Provides implementations of [[DataProduct]] for any [[Tuple2]] where each field has an implementation of `DataProduct`. */
  implicit def tuple2DataProduct[A : DataProduct, B : DataProduct]: DataProduct[(A, B)] = new DataProduct[(A, B)] {
    def dataIterator(tup: (A, B), path: String): Iterator[(Data, String)] = {
      val dpa = implicitly[DataProduct[A]]
      val dpb = implicitly[DataProduct[B]]
      val (a, b) = tup
      dpa.dataIterator(a, s"$path._1") ++ dpb.dataIterator(b, s"$path._2")
    }
  }

  /** Implicit conversion between `(A, B)` and `HWTuple2[A, B]` */
  implicit def tuple2hwtuple[T1 : DataProduct, T2 : DataProduct, V1 <: Data, V2 <: Data](
    tup: (T1, T2)
  )(
    implicit v1: DataView[T1, V1], v2: DataView[T2, V2]
  ): HWTuple2[V1, V2] = {
    tup.viewAs[HWTuple2[V1, V2]]
  }
}

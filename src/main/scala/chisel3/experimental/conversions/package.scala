
package chisel3.experimental

import chisel3._
import chisel3.experimental.dataview._
import scala.language.implicitConversions

/** Implicit conversions from some Scala standard library types and [[Data]]
  *
  * @note As this leans heavily on the experimental [[DataView]] feature, these APIs are experimental and subject to change
  */
package object conversions {

  /** Implicit conversion between `Seq` and `Vec` */
  implicit def seq2vec[A : DataProduct, B <: Data](xs: Seq[A])(implicit dv: DataView[A, B]): Vec[B] =
    xs.viewAs[Vec[B]]

  /** Implicit conversion between `(A, B)` and `HWTuple2[A, B]` */
  implicit def tuple2hwtuple[T1 : DataProduct, T2 : DataProduct, V1 <: Data, V2 <: Data](
    tup: (T1, T2)
  )(
    implicit v1: DataView[T1, V1], v2: DataView[T2, V2]
  ): HWTuple2[V1, V2] = {
    tup.viewAs[HWTuple2[V1, V2]]
  }
}

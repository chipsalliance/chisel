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
  implicit def seq2vec[A: DataProduct, B <: Data](xs: Seq[A])(implicit dv: DataView[A, B]): Vec[B] =
    xs.viewAs[Vec[B]]

  /** Implicit conversion between [[Tuple2]] and [[HWTuple2]] */
  implicit def tuple2hwtuple[T1: DataProduct, T2: DataProduct, V1 <: Data, V2 <: Data](
    tup: (T1, T2)
  )(
    implicit v1: DataView[T1, V1],
    v2:          DataView[T2, V2]
  ): HWTuple2[V1, V2] = {
    tup.viewAs[HWTuple2[V1, V2]]
  }

  /** Implicit conversion between [[Tuple3]] and [[HWTuple3]] */
  implicit def tuple3hwtuple[T1: DataProduct, T2: DataProduct, T3: DataProduct, V1 <: Data, V2 <: Data, V3 <: Data](
    tup: (T1, T2, T3)
  )(
    implicit v1: DataView[T1, V1],
    v2:          DataView[T2, V2],
    v3:          DataView[T3, V3]
  ): HWTuple3[V1, V2, V3] = {
    tup.viewAs[HWTuple3[V1, V2, V3]]
  }

  /** Implicit conversion between [[Tuple4]] and [[HWTuple4]] */
  implicit def tuple4hwtuple[
    T1: DataProduct,
    T2: DataProduct,
    T3: DataProduct,
    T4: DataProduct,
    V1 <: Data,
    V2 <: Data,
    V3 <: Data,
    V4 <: Data
  ](tup: (T1, T2, T3, T4)
  )(
    implicit v1: DataView[T1, V1],
    v2:          DataView[T2, V2],
    v3:          DataView[T3, V3],
    v4:          DataView[T4, V4]
  ): HWTuple4[V1, V2, V3, V4] = {
    tup.viewAs[HWTuple4[V1, V2, V3, V4]]
  }

  /** Implicit conversion between [[Tuple5]] and [[HWTuple5]] */
  implicit def tuple5hwtuple[
    T1: DataProduct,
    T2: DataProduct,
    T3: DataProduct,
    T4: DataProduct,
    T5: DataProduct,
    V1 <: Data,
    V2 <: Data,
    V3 <: Data,
    V4 <: Data,
    V5 <: Data
  ](tup: (T1, T2, T3, T4, T5)
  )(
    implicit v1: DataView[T1, V1],
    v2:          DataView[T2, V2],
    v3:          DataView[T3, V3],
    v4:          DataView[T4, V4],
    v5:          DataView[T5, V5]
  ): HWTuple5[V1, V2, V3, V4, V5] = {
    tup.viewAs[HWTuple5[V1, V2, V3, V4, V5]]
  }

  /** Implicit conversion between [[Tuple6]] and [[HWTuple6]] */
  implicit def tuple6hwtuple[
    T1: DataProduct,
    T2: DataProduct,
    T3: DataProduct,
    T4: DataProduct,
    T5: DataProduct,
    T6: DataProduct,
    V1 <: Data,
    V2 <: Data,
    V3 <: Data,
    V4 <: Data,
    V5 <: Data,
    V6 <: Data
  ](tup: (T1, T2, T3, T4, T5, T6)
  )(
    implicit v1: DataView[T1, V1],
    v2:          DataView[T2, V2],
    v3:          DataView[T3, V3],
    v4:          DataView[T4, V4],
    v5:          DataView[T5, V5],
    v6:          DataView[T6, V6]
  ): HWTuple6[V1, V2, V3, V4, V5, V6] = {
    tup.viewAs[HWTuple6[V1, V2, V3, V4, V5, V6]]
  }

  /** Implicit conversion between [[Tuple7]] and [[HWTuple7]] */
  implicit def tuple7hwtuple[
    T1: DataProduct,
    T2: DataProduct,
    T3: DataProduct,
    T4: DataProduct,
    T5: DataProduct,
    T6: DataProduct,
    T7: DataProduct,
    V1 <: Data,
    V2 <: Data,
    V3 <: Data,
    V4 <: Data,
    V5 <: Data,
    V6 <: Data,
    V7 <: Data
  ](tup: (T1, T2, T3, T4, T5, T6, T7)
  )(
    implicit v1: DataView[T1, V1],
    v2:          DataView[T2, V2],
    v3:          DataView[T3, V3],
    v4:          DataView[T4, V4],
    v5:          DataView[T5, V5],
    v6:          DataView[T6, V6],
    v7:          DataView[T7, V7]
  ): HWTuple7[V1, V2, V3, V4, V5, V6, V7] = {
    tup.viewAs[HWTuple7[V1, V2, V3, V4, V5, V6, V7]]
  }

  /** Implicit conversion between [[Tuple8]] and [[HWTuple8]] */
  implicit def tuple8hwtuple[
    T1: DataProduct,
    T2: DataProduct,
    T3: DataProduct,
    T4: DataProduct,
    T5: DataProduct,
    T6: DataProduct,
    T7: DataProduct,
    T8: DataProduct,
    V1 <: Data,
    V2 <: Data,
    V3 <: Data,
    V4 <: Data,
    V5 <: Data,
    V6 <: Data,
    V7 <: Data,
    V8 <: Data
  ](tup: (T1, T2, T3, T4, T5, T6, T7, T8)
  )(
    implicit v1: DataView[T1, V1],
    v2:          DataView[T2, V2],
    v3:          DataView[T3, V3],
    v4:          DataView[T4, V4],
    v5:          DataView[T5, V5],
    v6:          DataView[T6, V6],
    v7:          DataView[T7, V7],
    v8:          DataView[T8, V8]
  ): HWTuple8[V1, V2, V3, V4, V5, V6, V7, V8] = {
    tup.viewAs[HWTuple8[V1, V2, V3, V4, V5, V6, V7, V8]]
  }

  /** Implicit conversion between [[Tuple9]] and [[HWTuple9]] */
  implicit def tuple9hwtuple[
    T1: DataProduct,
    T2: DataProduct,
    T3: DataProduct,
    T4: DataProduct,
    T5: DataProduct,
    T6: DataProduct,
    T7: DataProduct,
    T8: DataProduct,
    T9: DataProduct,
    V1 <: Data,
    V2 <: Data,
    V3 <: Data,
    V4 <: Data,
    V5 <: Data,
    V6 <: Data,
    V7 <: Data,
    V8 <: Data,
    V9 <: Data
  ](tup: (T1, T2, T3, T4, T5, T6, T7, T8, T9)
  )(
    implicit v1: DataView[T1, V1],
    v2:          DataView[T2, V2],
    v3:          DataView[T3, V3],
    v4:          DataView[T4, V4],
    v5:          DataView[T5, V5],
    v6:          DataView[T6, V6],
    v7:          DataView[T7, V7],
    v8:          DataView[T8, V8],
    v9:          DataView[T9, V9]
  ): HWTuple9[V1, V2, V3, V4, V5, V6, V7, V8, V9] = {
    tup.viewAs[HWTuple9[V1, V2, V3, V4, V5, V6, V7, V8, V9]]
  }

  /** Implicit conversion between [[Tuple10]] and [[HWTuple10]] */
  implicit def tuple10hwtuple[
    T1:  DataProduct,
    T2:  DataProduct,
    T3:  DataProduct,
    T4:  DataProduct,
    T5:  DataProduct,
    T6:  DataProduct,
    T7:  DataProduct,
    T8:  DataProduct,
    T9:  DataProduct,
    T10: DataProduct,
    V1 <: Data,
    V2 <: Data,
    V3 <: Data,
    V4 <: Data,
    V5 <: Data,
    V6 <: Data,
    V7 <: Data,
    V8 <: Data,
    V9 <: Data,
    V10 <: Data
  ](tup: (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10)
  )(
    implicit v1: DataView[T1, V1],
    v2:          DataView[T2, V2],
    v3:          DataView[T3, V3],
    v4:          DataView[T4, V4],
    v5:          DataView[T5, V5],
    v6:          DataView[T6, V6],
    v7:          DataView[T7, V7],
    v8:          DataView[T8, V8],
    v9:          DataView[T9, V9],
    v10:         DataView[T10, V10]
  ): HWTuple10[V1, V2, V3, V4, V5, V6, V7, V8, V9, V10] = {
    tup.viewAs[HWTuple10[V1, V2, V3, V4, V5, V6, V7, V8, V9, V10]]
  }
}

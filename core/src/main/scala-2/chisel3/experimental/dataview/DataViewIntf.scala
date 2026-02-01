// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.dataview

import chisel3._
import chisel3.experimental.{HWTuple10, HWTuple2, HWTuple3, HWTuple4, HWTuple5, HWTuple6, HWTuple7, HWTuple8, HWTuple9}
import chisel3.experimental.{ChiselSubtypeOf, SourceInfo}

private[chisel3] trait DataViewIntf[T, V <: Data] { self: DataView[T, V] =>

  /** Compose two `DataViews` together to construct a view from the target of this `DataView` to the
    * view type of the second `DataView`
    *
    * @param g a DataView from `V` to new view-type `V2`
    * @tparam V2 View type of `DataView` `g`
    * @return a new `DataView` from the original `T` to new view-type `V2`
    */
  def andThen[V2 <: Data](g: DataView[V, V2])(implicit sourceInfo: SourceInfo): DataView[T, V2] =
    _andThenImpl(g)
}

private[chisel3] trait DataView$ObjIntf { self: DataView.type =>

  /** Default factory method, alias for [[pairs]] */
  def apply[T: DataProduct, V <: Data](
    mkView: T => V,
    pairs:  ((T, V) => (Data, Data))*
  )(
    implicit sourceInfo: SourceInfo
  ): DataView[T, V] = _applyImpl(mkView, pairs: _*)

  /** Construct [[DataView]]s with pairs of functions from the target and view to corresponding fields */
  def pairs[T: DataProduct, V <: Data](
    mkView: T => V,
    pairs:  ((T, V) => (Data, Data))*
  )(
    implicit sourceInfo: SourceInfo
  ): DataView[T, V] = _pairsImpl(mkView, pairs: _*)

  /** More general factory method for complex mappings */
  def mapping[T: DataProduct, V <: Data](
    mkView:  T => V,
    mapping: (T, V) => Iterable[(Data, Data)]
  )(
    implicit sourceInfo: SourceInfo
  ): DataView[T, V] = _mappingImpl(mkView, mapping)

  /** All Chisel Data are viewable as their own type */
  implicit def identityView[A <: Data](implicit sourceInfo: SourceInfo): DataView[A, A] =
    _identityViewImpl[A]

  /** Provides `DataView[Seq[A], Vec[B]]` for all `A` such that there exists `DataView[A, B]` */
  implicit def seqDataView[A: DataProduct, B <: Data](
    implicit dv: DataView[A, B],
    sourceInfo:  SourceInfo
  ): DataView[Seq[A], Vec[B]] = _seqDataViewImpl[A, B]

  /** Provides implementations of [[DataView]] for [[scala.Tuple2]] to [[HWTuple2]] */
  implicit def tuple2DataView[T1: DataProduct, T2: DataProduct, V1 <: Data, V2 <: Data](
    implicit v1: DataView[T1, V1],
    v2:          DataView[T2, V2],
    sourceInfo:  SourceInfo
  ): DataView[(T1, T2), HWTuple2[V1, V2]] = _tuple2DataViewImpl[T1, T2, V1, V2]

  /** Provides implementations of [[DataView]] for [[scala.Tuple3]] to [[HWTuple3]] */
  implicit def tuple3DataView[T1: DataProduct, T2: DataProduct, T3: DataProduct, V1 <: Data, V2 <: Data, V3 <: Data](
    implicit v1: DataView[T1, V1],
    v2:          DataView[T2, V2],
    v3:          DataView[T3, V3],
    sourceInfo:  SourceInfo
  ): DataView[(T1, T2, T3), HWTuple3[V1, V2, V3]] = _tuple3DataViewImpl[T1, T2, T3, V1, V2, V3]

  /** Provides implementations of [[DataView]] for [[scala.Tuple4]] to [[HWTuple4]] */
  implicit def tuple4DataView[
    T1: DataProduct,
    T2: DataProduct,
    T3: DataProduct,
    T4: DataProduct,
    V1 <: Data,
    V2 <: Data,
    V3 <: Data,
    V4 <: Data
  ](
    implicit v1: DataView[T1, V1],
    v2:          DataView[T2, V2],
    v3:          DataView[T3, V3],
    v4:          DataView[T4, V4],
    sourceInfo:  SourceInfo
  ): DataView[(T1, T2, T3, T4), HWTuple4[V1, V2, V3, V4]] =
    _tuple4DataViewImpl[T1, T2, T3, T4, V1, V2, V3, V4]

  /** Provides implementations of [[DataView]] for [[scala.Tuple5]] to [[HWTuple5]] */
  implicit def tuple5DataView[
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
  ](
    implicit v1: DataView[T1, V1],
    v2:          DataView[T2, V2],
    v3:          DataView[T3, V3],
    v4:          DataView[T4, V4],
    v5:          DataView[T5, V5],
    sourceInfo:  SourceInfo
  ): DataView[(T1, T2, T3, T4, T5), HWTuple5[V1, V2, V3, V4, V5]] =
    _tuple5DataViewImpl[T1, T2, T3, T4, T5, V1, V2, V3, V4, V5]

  /** Provides implementations of [[DataView]] for [[scala.Tuple6]] to [[HWTuple6]] */
  implicit def tuple6DataView[
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
  ](
    implicit v1: DataView[T1, V1],
    v2:          DataView[T2, V2],
    v3:          DataView[T3, V3],
    v4:          DataView[T4, V4],
    v5:          DataView[T5, V5],
    v6:          DataView[T6, V6],
    sourceInfo:  SourceInfo
  ): DataView[(T1, T2, T3, T4, T5, T6), HWTuple6[V1, V2, V3, V4, V5, V6]] =
    _tuple6DataViewImpl[T1, T2, T3, T4, T5, T6, V1, V2, V3, V4, V5, V6]

  /** Provides implementations of [[DataView]] for [[scala.Tuple7]] to [[HWTuple7]] */
  implicit def tuple7DataView[
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
  ](
    implicit v1: DataView[T1, V1],
    v2:          DataView[T2, V2],
    v3:          DataView[T3, V3],
    v4:          DataView[T4, V4],
    v5:          DataView[T5, V5],
    v6:          DataView[T6, V6],
    v7:          DataView[T7, V7],
    sourceInfo:  SourceInfo
  ): DataView[(T1, T2, T3, T4, T5, T6, T7), HWTuple7[V1, V2, V3, V4, V5, V6, V7]] =
    _tuple7DataViewImpl[T1, T2, T3, T4, T5, T6, T7, V1, V2, V3, V4, V5, V6, V7]

  /** Provides implementations of [[DataView]] for [[scala.Tuple8]] to [[HWTuple8]] */
  implicit def tuple8DataView[
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
  ](
    implicit v1: DataView[T1, V1],
    v2:          DataView[T2, V2],
    v3:          DataView[T3, V3],
    v4:          DataView[T4, V4],
    v5:          DataView[T5, V5],
    v6:          DataView[T6, V6],
    v7:          DataView[T7, V7],
    v8:          DataView[T8, V8],
    sourceInfo:  SourceInfo
  ): DataView[(T1, T2, T3, T4, T5, T6, T7, T8), HWTuple8[V1, V2, V3, V4, V5, V6, V7, V8]] =
    _tuple8DataViewImpl[T1, T2, T3, T4, T5, T6, T7, T8, V1, V2, V3, V4, V5, V6, V7, V8]

  /** Provides implementations of [[DataView]] for [[scala.Tuple9]] to [[HWTuple9]] */
  implicit def tuple9DataView[
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
  ](
    implicit v1: DataView[T1, V1],
    v2:          DataView[T2, V2],
    v3:          DataView[T3, V3],
    v4:          DataView[T4, V4],
    v5:          DataView[T5, V5],
    v6:          DataView[T6, V6],
    v7:          DataView[T7, V7],
    v8:          DataView[T8, V8],
    v9:          DataView[T9, V9],
    sourceInfo:  SourceInfo
  ): DataView[(T1, T2, T3, T4, T5, T6, T7, T8, T9), HWTuple9[V1, V2, V3, V4, V5, V6, V7, V8, V9]] =
    _tuple9DataViewImpl[T1, T2, T3, T4, T5, T6, T7, T8, T9, V1, V2, V3, V4, V5, V6, V7, V8, V9]

  /** Provides implementations of [[DataView]] for [[scala.Tuple10]] to [[HWTuple10]] */
  implicit def tuple10DataView[
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
  ](
    implicit v1: DataView[T1, V1],
    v2:          DataView[T2, V2],
    v3:          DataView[T3, V3],
    v4:          DataView[T4, V4],
    v5:          DataView[T5, V5],
    v6:          DataView[T6, V6],
    v7:          DataView[T7, V7],
    v8:          DataView[T8, V8],
    v9:          DataView[T9, V9],
    v10:         DataView[T10, V10],
    sourceInfo:  SourceInfo
  ): DataView[(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10), HWTuple10[V1, V2, V3, V4, V5, V6, V7, V8, V9, V10]] =
    _tuple10DataViewImpl[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10]
}

private[chisel3] trait PartialDataView$Intf { self: PartialDataView.type =>

  /** Default factory method, alias for [[pairs]] */
  def apply[T: DataProduct, V <: Data](
    mkView: T => V,
    pairs:  ((T, V) => (Data, Data))*
  )(
    implicit sourceInfo: SourceInfo
  ): DataView[T, V] = _applyImpl(mkView, pairs: _*)

  /** Construct [[DataView]]s with pairs of functions from the target and view to corresponding fields */
  def pairs[T: DataProduct, V <: Data](
    mkView: T => V,
    pairs:  ((T, V) => (Data, Data))*
  )(
    implicit sourceInfo: SourceInfo
  ): DataView[T, V] = _pairsImpl(mkView, pairs: _*)

  /** More general factory method for complex mappings */
  def mapping[T: DataProduct, V <: Data](
    mkView:  T => V,
    mapping: (T, V) => Iterable[(Data, Data)]
  )(
    implicit sourceInfo: SourceInfo
  ): DataView[T, V] = _mappingImpl(mkView, mapping)

  /** Constructs a non-total [[DataView]] mapping from a [[Bundle]] or [[Record]] type to a parent [[Bundle]] or [[Record]] type
    *
    * @param mkView a function constructing an instance `V` from an instance of `T`
    * @return the [[DataView]] that enables viewing instances of a [[Bundle]]/[[Record]] as instances of a parent type
    */
  def supertype[T <: Record, V <: Record](
    mkView: T => V
  )(
    implicit ev: ChiselSubtypeOf[T, V],
    sourceInfo:  SourceInfo
  ): DataView[T, V] = _supertypeImpl(mkView)
}

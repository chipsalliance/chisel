// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.dataview

import chisel3._
import chisel3.reflect.DataMirror.internal.chiselTypeClone
import chisel3.experimental.{HWTuple10, HWTuple2, HWTuple3, HWTuple4, HWTuple5, HWTuple6, HWTuple7, HWTuple8, HWTuple9}
import chisel3.experimental.{ChiselSubtypeOf, SourceInfo, UnlocatableSourceInfo}

import scala.reflect.runtime.universe.WeakTypeTag
import annotation.implicitNotFound

/** Mapping between a target type `T` and a view type `V`
  *
  * Enables calling `.viewAs[T]` on instances of the target type.
  *
  * ==Detailed documentation==
  *   - [[https://www.chisel-lang.org/chisel3/docs/explanations/dataview Explanation]]
  *   - [[https://www.chisel-lang.org/chisel3/docs/cookbooks/dataview Cookbook]]
  *
  * @example {{{
  * class Foo(val w: Int) extends Bundle {
  *   val a = UInt(w.W)
  * }
  * class Bar(val w: Int) extends Bundle {
  *   val b = UInt(w.W)
  * }
  * // DataViews are created using factory methods in the companion object
  * implicit val view = DataView[Foo, Bar](
  *   // The first argument is a function constructing a Foo from a Bar
  *   foo => new Bar(foo.w)
  *   // The remaining arguments are a variable number of field pairings
  *   _.a -> _.b
  * )
  * }}}
  *
  * @tparam T Target type (must have an implementation of [[DataProduct]])
  * @tparam V View type
  * @see [[DataView$ object DataView]] for factory methods
  * @see [[PartialDataView object PartialDataView]] for defining non-total `DataViews`
  */
@implicitNotFound(
  "Could not find implicit value for DataView[${T}, ${V}].\n" +
    "Please see https://www.chisel-lang.org/chisel3/docs/explanations/dataview"
)
sealed class DataView[T: DataProduct, V <: Data] private[chisel3] (
  /** Function constructing an object of the View type from an object of the Target type */
  private[chisel3] val mkView: T => V,
  /** Function that returns corresponding fields of the target and view */
  private[chisel3] val mapping: (T, V) => Iterable[(Data, Data)],
  // Aliasing this with a def below to make the ScalaDoc show up for the field
  _total: Boolean
)(
  implicit private[chisel3] val sourceInfo: SourceInfo) {

  /** Indicates if the mapping contains every field of the target */
  def total: Boolean = _total

  override def toString: String = {
    val base = sourceInfo.makeMessage(x => x)
    val loc = if (base.nonEmpty) base else "@unknown"
    val name = if (total) "DataView" else "PartialDataView"
    s"$name(defined $loc)"
  }

  /** Compose two `DataViews` together to construct a view from the target of this `DataView` to the
    * view type of the second `DataView`
    *
    * @param g a DataView from `V` to new view-type `V2`
    * @tparam V2 View type of `DataView` `g`
    * @return a new `DataView` from the original `T` to new view-type `V2`
    */
  def andThen[V2 <: Data](g: DataView[V, V2])(implicit sourceInfo: SourceInfo): DataView[T, V2] = {
    implicit val self: DataView[T, V] = this
    implicit val gg:   DataView[V, V2] = g
    new DataView[T, V2](
      t => g.mkView(mkView(t)),
      { case (t, v2) => List(t.viewAs[V].viewAs[V2] -> v2) },
      this.total && g.total
    ) {
      override def toString: String = s"$self andThen $g"
    }
  }
}

/** Factory methods for constructing [[DataView]]s, see class for example use */
object DataView {

  /** Default factory method, alias for [[pairs]] */
  def apply[T: DataProduct, V <: Data](
    mkView: T => V,
    pairs:  ((T, V) => (Data, Data))*
  )(
    implicit sourceInfo: SourceInfo
  ): DataView[T, V] =
    DataView.pairs(mkView, pairs: _*)

  /** Construct [[DataView]]s with pairs of functions from the target and view to corresponding fields */
  def pairs[T: DataProduct, V <: Data](
    mkView: T => V,
    pairs:  ((T, V) => (Data, Data))*
  )(
    implicit sourceInfo: SourceInfo
  ): DataView[T, V] =
    mapping(mkView: T => V, swizzle(pairs))

  /** More general factory method for complex mappings */
  def mapping[T: DataProduct, V <: Data](
    mkView:  T => V,
    mapping: (T, V) => Iterable[(Data, Data)]
  )(
    implicit sourceInfo: SourceInfo
  ): DataView[T, V] =
    new DataView[T, V](mkView, mapping, _total = true)

  /** Provides `invert` for invertible [[DataView]]s
    *
    * This must be done as an extension method because it applies an addition constraint on the `Target`
    * type parameter, namely that it must be a subtype of [[Data]].
    *
    * @note [[PartialDataView]]s are **not** invertible and will result in an elaboration time exception
    */
  implicit class InvertibleDataView[T <: Data: WeakTypeTag, V <: Data: WeakTypeTag](view: DataView[T, V]) {
    def invert(mkView: V => T): DataView[V, T] = {
      // It would've been nice to make this a compiler error, but it's unclear how to make that work.
      // We tried having separate TotalDataView and PartialDataView and only defining inversion for
      // TotalDataView. For some reason, implicit resolution wouldn't invert TotalDataViews. This is
      // probably because it was looking for the super-type DataView and since invertDataView was
      // only defined on TotalDataView, it wasn't included in implicit resolution. Thus we end up
      // with a runtime check.
      if (!view.total) {
        val tt = implicitly[WeakTypeTag[T]].tpe
        val vv = implicitly[WeakTypeTag[V]].tpe
        val msg = s"Cannot invert '$view' as it is non-total.\n  Try providing a DataView[$vv, $tt]." +
          s"\n  Please see https://www.chisel-lang.org/chisel3/docs/explanations/dataview."
        throw InvalidViewException(msg)
      }
      implicit val sourceInfo = view.sourceInfo
      new DataView[V, T](mkView, swapArgs(view.mapping), view.total)
    }
  }

  private[dataview] def swizzle[A, B, C, D](fs: Iterable[(A, B) => (C, D)]): (A, B) => Iterable[(C, D)] = {
    case (a, b) => fs.map(f => f(a, b))
  }

  private def swapArgs[A, B, C, D](f: (A, B) => Iterable[(C, D)]): (B, A) => Iterable[(D, C)] = {
    case (b, a) => f(a, b).map(_.swap)
  }

  // ****************************** Built-in Implementations of DataView ******************************
  // Sort of the "Standard library" implementations

  /** All Chisel Data are viewable as their own type */
  implicit def identityView[A <: Data](implicit sourceInfo: SourceInfo): DataView[A, A] =
    DataView[A, A](chiselTypeOf.apply, { case (x, y) => (x, y) })

  /** Provides `DataView[Seq[A], Vec[B]]` for all `A` such that there exists `DataView[A, B]` */
  implicit def seqDataView[A: DataProduct, B <: Data](
    implicit dv: DataView[A, B],
    sourceInfo:  SourceInfo
  ): DataView[Seq[A], Vec[B]] = {
    // TODO this would need a better way to determine the prototype for the Vec
    DataView.mapping[Seq[A], Vec[B]](
      xs => Vec(xs.size, chiselTypeClone(xs.head.viewAs[B]))(sourceInfo), // xs.head is not correct in general
      { case (s, v) => s.zip(v).map { case (a, b) => a.viewAs[B] -> b } }
    )
  }

  /** Provides implementations of [[DataView]] for [[scala.Tuple2]]  to [[HWTuple2]] */
  implicit def tuple2DataView[T1: DataProduct, T2: DataProduct, V1 <: Data, V2 <: Data](
    implicit v1: DataView[T1, V1],
    v2:          DataView[T2, V2],
    sourceInfo:  SourceInfo
  ): DataView[(T1, T2), HWTuple2[V1, V2]] =
    DataView.mapping(
      { case (a, b) => new HWTuple2(a.viewAs[V1].cloneType, b.viewAs[V2].cloneType) },
      {
        case ((a, b), hwt) =>
          Seq(a.viewAs[V1] -> hwt._1, b.viewAs[V2] -> hwt._2)
      }
    )

  /** Provides implementations of [[DataView]] for [[scala.Tuple3]] to [[HWTuple3]] */
  implicit def tuple3DataView[T1: DataProduct, T2: DataProduct, T3: DataProduct, V1 <: Data, V2 <: Data, V3 <: Data](
    implicit v1: DataView[T1, V1],
    v2:          DataView[T2, V2],
    v3:          DataView[T3, V3],
    sourceInfo:  SourceInfo
  ): DataView[(T1, T2, T3), HWTuple3[V1, V2, V3]] =
    DataView.mapping(
      { case (a, b, c) => new HWTuple3(a.viewAs[V1].cloneType, b.viewAs[V2].cloneType, c.viewAs[V3].cloneType) },
      {
        case ((a, b, c), hwt) =>
          Seq(a.viewAs[V1] -> hwt._1, b.viewAs[V2] -> hwt._2, c.viewAs[V3] -> hwt._3)
      }
    )

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
    DataView.mapping(
      {
        case (a, b, c, d) =>
          new HWTuple4(a.viewAs[V1].cloneType, b.viewAs[V2].cloneType, c.viewAs[V3].cloneType, d.viewAs[V4].cloneType)
      },
      {
        case ((a, b, c, d), hwt) =>
          Seq(a.viewAs[V1] -> hwt._1, b.viewAs[V2] -> hwt._2, c.viewAs[V3] -> hwt._3, d.viewAs[V4] -> hwt._4)
      }
    )

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
  ): DataView[(T1, T2, T3, T4, T5), HWTuple5[V1, V2, V3, V4, V5]] = {
    DataView.mapping(
      {
        case tup: Tuple5[T1, T2, T3, T4, T5] =>
          val (a, b, c, d, e) = tup
          new HWTuple5(
            a.viewAs[V1].cloneType,
            b.viewAs[V2].cloneType,
            c.viewAs[V3].cloneType,
            d.viewAs[V4].cloneType,
            e.viewAs[V5].cloneType
          )
      },
      {
        case ((a, b, c, d, e), hwt) =>
          Seq(
            a.viewAs[V1] -> hwt._1,
            b.viewAs[V2] -> hwt._2,
            c.viewAs[V3] -> hwt._3,
            d.viewAs[V4] -> hwt._4,
            e.viewAs[V5] -> hwt._5
          )
      }
    )
  }

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
    DataView.mapping(
      {
        case (a, b, c, d, e, f) =>
          new HWTuple6(
            a.viewAs[V1].cloneType,
            b.viewAs[V2].cloneType,
            c.viewAs[V3].cloneType,
            d.viewAs[V4].cloneType,
            e.viewAs[V5].cloneType,
            f.viewAs[V6].cloneType
          )
      },
      {
        case ((a, b, c, d, e, f), hwt) =>
          Seq(
            a.viewAs[V1] -> hwt._1,
            b.viewAs[V2] -> hwt._2,
            c.viewAs[V3] -> hwt._3,
            d.viewAs[V4] -> hwt._4,
            e.viewAs[V5] -> hwt._5,
            f.viewAs[V6] -> hwt._6
          )
      }
    )

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
    DataView.mapping(
      {
        case (a, b, c, d, e, f, g) =>
          new HWTuple7(
            a.viewAs[V1].cloneType,
            b.viewAs[V2].cloneType,
            c.viewAs[V3].cloneType,
            d.viewAs[V4].cloneType,
            e.viewAs[V5].cloneType,
            f.viewAs[V6].cloneType,
            g.viewAs[V7].cloneType
          )
      },
      {
        case ((a, b, c, d, e, f, g), hwt) =>
          Seq(
            a.viewAs[V1] -> hwt._1,
            b.viewAs[V2] -> hwt._2,
            c.viewAs[V3] -> hwt._3,
            d.viewAs[V4] -> hwt._4,
            e.viewAs[V5] -> hwt._5,
            f.viewAs[V6] -> hwt._6,
            g.viewAs[V7] -> hwt._7
          )
      }
    )

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
    DataView.mapping(
      {
        case (a, b, c, d, e, f, g, h) =>
          new HWTuple8(
            a.viewAs[V1].cloneType,
            b.viewAs[V2].cloneType,
            c.viewAs[V3].cloneType,
            d.viewAs[V4].cloneType,
            e.viewAs[V5].cloneType,
            f.viewAs[V6].cloneType,
            g.viewAs[V7].cloneType,
            h.viewAs[V8].cloneType
          )
      },
      {
        case ((a, b, c, d, e, f, g, h), hwt) =>
          Seq(
            a.viewAs[V1] -> hwt._1,
            b.viewAs[V2] -> hwt._2,
            c.viewAs[V3] -> hwt._3,
            d.viewAs[V4] -> hwt._4,
            e.viewAs[V5] -> hwt._5,
            f.viewAs[V6] -> hwt._6,
            g.viewAs[V7] -> hwt._7,
            h.viewAs[V8] -> hwt._8
          )
      }
    )

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
    DataView.mapping(
      {
        case (a, b, c, d, e, f, g, h, i) =>
          new HWTuple9(
            a.viewAs[V1].cloneType,
            b.viewAs[V2].cloneType,
            c.viewAs[V3].cloneType,
            d.viewAs[V4].cloneType,
            e.viewAs[V5].cloneType,
            f.viewAs[V6].cloneType,
            g.viewAs[V7].cloneType,
            h.viewAs[V8].cloneType,
            i.viewAs[V9].cloneType
          )
      },
      {
        case ((a, b, c, d, e, f, g, h, i), hwt) =>
          Seq(
            a.viewAs[V1] -> hwt._1,
            b.viewAs[V2] -> hwt._2,
            c.viewAs[V3] -> hwt._3,
            d.viewAs[V4] -> hwt._4,
            e.viewAs[V5] -> hwt._5,
            f.viewAs[V6] -> hwt._6,
            g.viewAs[V7] -> hwt._7,
            h.viewAs[V8] -> hwt._8,
            i.viewAs[V9] -> hwt._9
          )
      }
    )

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
    DataView.mapping(
      {
        case (a, b, c, d, e, f, g, h, i, j) =>
          new HWTuple10(
            a.viewAs[V1].cloneType,
            b.viewAs[V2].cloneType,
            c.viewAs[V3].cloneType,
            d.viewAs[V4].cloneType,
            e.viewAs[V5].cloneType,
            f.viewAs[V6].cloneType,
            g.viewAs[V7].cloneType,
            h.viewAs[V8].cloneType,
            i.viewAs[V9].cloneType,
            j.viewAs[V10].cloneType
          )
      },
      {
        case ((a, b, c, d, e, f, g, h, i, j), hwt) =>
          Seq(
            a.viewAs[V1] -> hwt._1,
            b.viewAs[V2] -> hwt._2,
            c.viewAs[V3] -> hwt._3,
            d.viewAs[V4] -> hwt._4,
            e.viewAs[V5] -> hwt._5,
            f.viewAs[V6] -> hwt._6,
            g.viewAs[V7] -> hwt._7,
            h.viewAs[V8] -> hwt._8,
            i.viewAs[V9] -> hwt._9,
            j.viewAs[V10] -> hwt._10
          )
      }
    )
}

/** Factory methods for constructing non-total [[DataView]]s */
object PartialDataView {

  /** Default factory method, alias for [[pairs]] */
  def apply[T: DataProduct, V <: Data](
    mkView: T => V,
    pairs:  ((T, V) => (Data, Data))*
  )(
    implicit sourceInfo: SourceInfo
  ): DataView[T, V] =
    PartialDataView.pairs(mkView, pairs: _*)

  /** Construct [[DataView]]s with pairs of functions from the target and view to corresponding fields */
  def pairs[T: DataProduct, V <: Data](
    mkView: T => V,
    pairs:  ((T, V) => (Data, Data))*
  )(
    implicit sourceInfo: SourceInfo
  ): DataView[T, V] =
    mapping(mkView, DataView.swizzle(pairs))

  /** More general factory method for complex mappings */
  def mapping[T: DataProduct, V <: Data](
    mkView:  T => V,
    mapping: (T, V) => Iterable[(Data, Data)]
  )(
    implicit sourceInfo: SourceInfo
  ): DataView[T, V] =
    new DataView[T, V](mkView, mapping, _total = false)

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
  ): DataView[T, V] =
    mapping[T, V](
      mkView,
      {
        case (a, b) =>
          val aElts = a._elements
          val bElts = b._elements
          val bKeys = bElts.keySet
          val keys = aElts.keysIterator.filter(bKeys.contains)
          keys.map(k => aElts(k) -> bElts(k)).toSeq
      }
    )
}

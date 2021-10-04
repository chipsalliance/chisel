// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.dataview

import chisel3._
import chisel3.internal.sourceinfo.SourceInfo
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
@implicitNotFound("Could not find implicit value for DataView[${T}, ${V}].\n" +
  "Please see https://www.chisel-lang.org/chisel3/docs/explanations/dataview")
sealed class DataView[T : DataProduct, V <: Data] private[chisel3] (
  /** Function constructing an object of the View type from an object of the Target type */
  private[chisel3] val mkView: T => V,
  /** Function that returns corresponding fields of the target and view */
  private[chisel3] val mapping: (T, V) => Iterable[(Data, Data)],
  // Aliasing this with a def below to make the ScalaDoc show up for the field
  _total: Boolean
)(
  implicit private[chisel3] val sourceInfo: SourceInfo
) {
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
    val self = this
    // We have to pass the DataProducts and DataViews manually to .viewAs below
    val tdp = implicitly[DataProduct[T]]
    val vdp = implicitly[DataProduct[V]]
    new DataView[T, V2](
      t => g.mkView(mkView(t)),
      { case (t, v2) => List(t.viewAs[V](tdp, self).viewAs[V2](vdp, g) -> v2) },
      this.total && g.total
    ) {
      override def toString: String = s"$self andThen $g"
    }
  }
}

/** Factory methods for constructing [[DataView]]s, see class for example use */
object DataView {

  /** Default factory method, alias for [[pairs]] */
  def apply[T : DataProduct, V <: Data](mkView: T => V, pairs: ((T, V) => (Data, Data))*)(implicit sourceInfo: SourceInfo): DataView[T, V] =
    DataView.pairs(mkView, pairs: _*)

  /** Construct [[DataView]]s with pairs of functions from the target and view to corresponding fields */
  def pairs[T : DataProduct, V <: Data](mkView: T => V, pairs: ((T, V) => (Data, Data))*)(implicit sourceInfo: SourceInfo): DataView[T, V] =
    mapping(mkView: T => V, swizzle(pairs))

  /** More general factory method for complex mappings */
  def mapping[T : DataProduct, V <: Data](mkView: T => V, mapping: (T, V) => Iterable[(Data, Data)])(implicit sourceInfo: SourceInfo): DataView[T, V] =
    new DataView[T, V](mkView, mapping, _total = true)

  /** Provides `invert` for invertible [[DataView]]s
    *
    * This must be done as an extension method because it applies an addition constraint on the `Target`
    * type parameter, namely that it must be a subtype of [[Data]].
    *
    * @note [[PartialDataView]]s are **not** invertible and will result in an elaboration time exception
    */
  implicit class InvertibleDataView[T <: Data : WeakTypeTag, V <: Data : WeakTypeTag](view: DataView[T, V]) {
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

  /** All Chisel Data are viewable as their own type */
  implicit def identityView[A <: Data](implicit sourceInfo: SourceInfo): DataView[A, A] =
    DataView[A, A](chiselTypeOf.apply, { case (x, y) => (x, y) })
}

/** Factory methods for constructing non-total [[DataView]]s */
object PartialDataView {

  /** Default factory method, alias for [[pairs]] */
  def apply[T: DataProduct, V <: Data](mkView: T => V, pairs: ((T, V) => (Data, Data))*)(implicit sourceInfo: SourceInfo): DataView[T, V] =
    PartialDataView.pairs(mkView, pairs: _*)

  /** Construct [[DataView]]s with pairs of functions from the target and view to corresponding fields */
  def pairs[T: DataProduct, V <: Data](mkView: T => V, pairs: ((T, V) => (Data, Data))*)(implicit sourceInfo: SourceInfo): DataView[T, V] =
    mapping(mkView, DataView.swizzle(pairs))

  /** More general factory method for complex mappings */
  def mapping[T: DataProduct, V <: Data](mkView: T => V, mapping: (T, V) => Iterable[(Data, Data)])(implicit sourceInfo: SourceInfo): DataView[T, V] =
    new DataView[T, V](mkView, mapping, _total = false)
}

package chisel3.experimental.dataview

import chisel3._
import scala.reflect.runtime.universe.WeakTypeTag

object InvertibleDataView {
  private def swapArgs[A, B, C, D](f: (A, B) => Iterable[(C, D)]): (B, A) => Iterable[(D, C)] = {
    case (b, a) => f(a, b).map(_.swap)
  }

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
}

// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.dataview

import chisel3._
import scala.reflect.ClassTag

private[chisel3] trait InvertibleDataView {
  def swapArgs[A, B, C, D](f: (A, B) => Iterable[(C, D)]): (B, A) => Iterable[(D, C)] = { case (b, a) =>
    f(a, b).map(_.swap)
  }

  /** Provides `invert` for invertible [[DataView]]s
    *
    * This must be done as an extension method because it applies an
    * addition constraint on the `Target` type parameter, namely that
    * it must be a subtype of [[Data]].
    *
    * @note [[PartialDataView]]s are **not** invertible and will
    * result in an elaboration time exception
    */
  implicit class InvertibleDataView[T <: Data: ClassTag, V <: Data: ClassTag](view: DataView[T, V]) {
    def invert(mkView: V => T): DataView[V, T] = {
      // Keep this a runtime error to align with Scala 2
      if (!view.total) {
        val tt = summon[ClassTag[T]].runtimeClass.getCanonicalName
        val vv = summon[ClassTag[V]].runtimeClass.getCanonicalName
        val msg = s"Cannot invert '$view' as it is non-total.\n  Try providing a DataView[$vv, $tt]." +
          s"\n  Please see https://www.chisel-lang.org/chisel3/docs/explanations/dataview."
        throw InvalidViewException(msg)
      }
      implicit val sourceInfo = view.sourceInfo
      new DataView[V, T](mkView, swapArgs(view.mapping), view.total)
    }
  }
}

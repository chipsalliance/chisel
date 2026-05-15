// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo

/** Internal helper for creating ports from values whose shape is described by a
  * [[chisel3.experimental.dataview.DataProduct]].
  *
  * For a single [[Data]] this defers to [[FlatIO]] so existing
  * `FixedIOBaseModule` behavior is preserved exactly. For composite shapes
  * (tuples, sequences) it recursively creates one port per contained [[Data]],
  * naming each port based on its structural path (e.g. `_1`, `_2`, `_0`).
  *
  * The original generator value cannot be reused as the bound IO because
  * [[IO]]/[[FlatIO]] may clone the provided [[Data]] (per
  * [[chisel3.Data.mustClone]]). Instead, this helper rebuilds the structure of
  * `A` with the cloned, port-bound values substituted in, returning a value of
  * the same shape whose leaves are the live ports.
  */
private[chisel3] object FixedIO {

  /** Bind `value`'s contained [[Data]] as ports, returning a value of the same
    * shape whose leaves are the bound ports.
    *
    * The chisel naming plugin would normally inject a `withName("io")` around
    * the val RHS for `final val io = ...`, but only when the val's type is
    * known to be a [[Data]] (or container thereof). Because `A` is an
    * unbounded type parameter, the plugin does not insert that wrapper, so we
    * must call [[chisel3.withName]] explicitly here.
    *
    * @param value the generator value to walk
    * @param path  the structural path used to name multi-IO leaves (the
    *              top-level call typically passes `"io"`)
    */
  def bindPorts(value: Any, path: String)(implicit si: SourceInfo): Any =
    chisel3.withName(path)(bindPortsImpl(value, path))

  private def bindPortsImpl(value: Any, path: String)(implicit si: SourceInfo): Any = value match {
    case d: Data =>
      // Top-level invocation with `path == "io"` and a single Data preserves
      // existing FixedIOBaseModule behavior exactly (uses FlatIO so Bundles
      // become flat top-level ports).
      if (path == "io") FlatIO(d)
      else {
        val port = IO(d)
        port.suggestName(path)
        port
      }
    case t: Tuple1[_] =>
      Tuple1(bindPorts(t._1, appendPath(path, "_1")))
    case t: Tuple2[_, _] =>
      (bindPorts(t._1, appendPath(path, "_1")), bindPorts(t._2, appendPath(path, "_2")))
    case t: Tuple3[_, _, _] =>
      (
        bindPorts(t._1, appendPath(path, "_1")),
        bindPorts(t._2, appendPath(path, "_2")),
        bindPorts(t._3, appendPath(path, "_3"))
      )
    case t: Tuple4[_, _, _, _] =>
      (
        bindPorts(t._1, appendPath(path, "_1")),
        bindPorts(t._2, appendPath(path, "_2")),
        bindPorts(t._3, appendPath(path, "_3")),
        bindPorts(t._4, appendPath(path, "_4"))
      )
    case t: Tuple5[_, _, _, _, _] =>
      (
        bindPorts(t._1, appendPath(path, "_1")),
        bindPorts(t._2, appendPath(path, "_2")),
        bindPorts(t._3, appendPath(path, "_3")),
        bindPorts(t._4, appendPath(path, "_4")),
        bindPorts(t._5, appendPath(path, "_5"))
      )
    case t: Tuple6[_, _, _, _, _, _] =>
      (
        bindPorts(t._1, appendPath(path, "_1")),
        bindPorts(t._2, appendPath(path, "_2")),
        bindPorts(t._3, appendPath(path, "_3")),
        bindPorts(t._4, appendPath(path, "_4")),
        bindPorts(t._5, appendPath(path, "_5")),
        bindPorts(t._6, appendPath(path, "_6"))
      )
    case t: Tuple7[_, _, _, _, _, _, _] =>
      (
        bindPorts(t._1, appendPath(path, "_1")),
        bindPorts(t._2, appendPath(path, "_2")),
        bindPorts(t._3, appendPath(path, "_3")),
        bindPorts(t._4, appendPath(path, "_4")),
        bindPorts(t._5, appendPath(path, "_5")),
        bindPorts(t._6, appendPath(path, "_6")),
        bindPorts(t._7, appendPath(path, "_7"))
      )
    case t: Tuple8[_, _, _, _, _, _, _, _] =>
      (
        bindPorts(t._1, appendPath(path, "_1")),
        bindPorts(t._2, appendPath(path, "_2")),
        bindPorts(t._3, appendPath(path, "_3")),
        bindPorts(t._4, appendPath(path, "_4")),
        bindPorts(t._5, appendPath(path, "_5")),
        bindPorts(t._6, appendPath(path, "_6")),
        bindPorts(t._7, appendPath(path, "_7")),
        bindPorts(t._8, appendPath(path, "_8"))
      )
    case t: Tuple9[_, _, _, _, _, _, _, _, _] =>
      (
        bindPorts(t._1, appendPath(path, "_1")),
        bindPorts(t._2, appendPath(path, "_2")),
        bindPorts(t._3, appendPath(path, "_3")),
        bindPorts(t._4, appendPath(path, "_4")),
        bindPorts(t._5, appendPath(path, "_5")),
        bindPorts(t._6, appendPath(path, "_6")),
        bindPorts(t._7, appendPath(path, "_7")),
        bindPorts(t._8, appendPath(path, "_8")),
        bindPorts(t._9, appendPath(path, "_9"))
      )
    case t: Tuple10[_, _, _, _, _, _, _, _, _, _] =>
      (
        bindPorts(t._1, appendPath(path, "_1")),
        bindPorts(t._2, appendPath(path, "_2")),
        bindPorts(t._3, appendPath(path, "_3")),
        bindPorts(t._4, appendPath(path, "_4")),
        bindPorts(t._5, appendPath(path, "_5")),
        bindPorts(t._6, appendPath(path, "_6")),
        bindPorts(t._7, appendPath(path, "_7")),
        bindPorts(t._8, appendPath(path, "_8")),
        bindPorts(t._9, appendPath(path, "_9")),
        bindPorts(t._10, appendPath(path, "_10"))
      )
    case s: Seq[_] =>
      s.zipWithIndex.map { case (elt, idx) => bindPorts(elt, appendPath(path, idx.toString)) }
    case other =>
      // Anything else (e.g. types with DataProduct.empty) has no Data to bind.
      other
  }

  private def appendPath(path: String, suffix: String): String =
    if (path.isEmpty) suffix else s"${path}_${suffix.stripPrefix("_")}"
}

// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

private[chisel3] trait attach$Intf { self: attach.type =>

  /** Create an electrical connection between [[Analog]] components
    *
    * @param elts The components to attach
    *
    * @example
    * {{{
    * val a1 = Wire(Analog(32.W))
    * val a2 = Wire(Analog(32.W))
    * attach(a1, a2)
    * }}}
    */
  def apply(elts: Analog*)(using SourceInfo): Unit = _applyImpl(elts: _*)
}

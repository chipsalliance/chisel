package chisel3

import chisel3.internal.NamedComponent

/** Provides an underlying target-able class for SRAM.
  */
private[chisel3] final class SramTarget() extends NamedComponent {
  _parent.foreach(_.addId(this))
}

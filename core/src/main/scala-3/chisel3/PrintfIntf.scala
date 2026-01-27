// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo

private[chisel3] trait PrintfIntf { self: printf.type =>
  def apply(fmt: String, data: Bits*)(implicit sourceInfo: SourceInfo): chisel3.printf.Printf =
    printf.apply(Printable.pack(fmt, data: _*))
}

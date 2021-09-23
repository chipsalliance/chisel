// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

import chisel3.{Bool, CompileOptions}
import chisel3.internal.sourceinfo.SourceInfo

package object verification {

  object assert {
    @deprecated("Please use chisel3.assert instead. The chisel3.experimental.verification package will be removed.", "Chisel 3.4")
    def apply(predicate: Bool, msg: String = "")
      (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): chisel3.assert.Assert = chisel3.assert(predicate, msg)
  }

  object assume {
    @deprecated("Please use chisel3.assume instead. The chisel3.experimental.verification package will be removed.", "Chisel 3.4")
    def apply(predicate: Bool, msg: String = "")
      (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): chisel3.assume.Assume = chisel3.assume(predicate, msg)
  }

  object cover {
    @deprecated("Please use chisel3.cover instead. The chisel3.experimental.verification package will be removed.", "Chisel 3.4")
    def apply(predicate: Bool, msg: String = "")
      (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): chisel3.cover.Cover = chisel3.cover(predicate, msg)
  }
}

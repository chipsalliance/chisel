// SPDX-License-Identifier: Apache-2.0

package chisel3.testers

import TesterDriver.Backend
import chisel3.{Bundle, RawModule}
import chisel3.internal.firrtl.Circuit
import chisel3.stage.ChiselStage
import firrtl.AnnotationSeq

object TestUtils {
  // Useful because TesterDriver.Backend is chisel3 package private
  def containsBackend(annos: AnnotationSeq): Boolean =
    annos.collectFirst { case b: Backend => b }.isDefined

  // Allows us to check that the compiler plugin cloneType is actually working
  val usingPlugin: Boolean = (new Bundle { def check = _usingPlugin }).check
  def elaborateNoReflectiveAutoCloneType(f: => RawModule): Circuit = {
    ChiselStage.elaborate {
      chisel3.internal.Builder.allowReflectiveAutoCloneType = !usingPlugin
      f
    }
  }
}

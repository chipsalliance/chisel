// SPDX-License-Identifier: Apache-2.0

package chiselTests.testers

import chisel3.internal.firrtl.Circuit
import chisel3.stage.ChiselStage
import chisel3.{Bundle, RawModule}
import chiselTests.testers.TesterDriver.Backend
import firrtl.AnnotationSeq

object TestUtils {
  // Useful because TesterDriver.Backend is chisel3 package private
  def containsBackend(annos: AnnotationSeq): Boolean =
    annos.collectFirst { case b: Backend => b }.isDefined

  // Allows us to check that the compiler plugin cloneType is actually working
  val usingPlugin: Boolean = (new Bundle { def check = _usingPlugin }).check
  def elaborateNoReflectiveAutoCloneType(f: => RawModule): Circuit = {
    ChiselStage.elaborate {
      chisel3.hack.Builder.allowReflectiveAutoCloneType = !usingPlugin
      f
    }
  }
}

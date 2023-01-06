// SPDX-License-Identifier: Apache-2.0

package chisel3.testers

import chisel3.testers.TesterDriver.Backend
import firrtl.AnnotationSeq

object TestUtils {
  // Useful because TesterDriver.Backend is chisel3 package private
  def containsBackend(annos: AnnotationSeq): Boolean =
    annos.collectFirst { case b: Backend => b }.isDefined
}

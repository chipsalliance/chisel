// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.experimental.minimizer
import chisel3.util.experimental.decode.Minimizer
import chisel3.util.experimental.decode.QMCMinimizer

class QMCSpec extends MinimizerSpec {
  override def minimizer: Minimizer = QMCMinimizer
}

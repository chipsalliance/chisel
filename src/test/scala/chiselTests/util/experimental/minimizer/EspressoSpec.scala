// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.experimental.minimizer
import chisel3.util.experimental.decode.EspressoMinimizer
import chisel3.util.experimental.decode.Minimizer

class EspressoSpec extends MinimizerSpec {
  override def minimizer: Minimizer = EspressoMinimizer
}

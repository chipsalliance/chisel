// SPDX-License-Identifier: Apache-2.0
package firrtl
package benchmark
package hot

import firrtl._
import firrtl.passes.LowerTypes
import firrtl.stage.TransformManager
import firrtl.benchmark.util._
import firrtl.transforms.DedupModules

abstract class TransformBenchmark(factory: () => Transform) extends App {
  val inputFile = args(0)
  val warmup = args(1).toInt
  val runs = args(2).toInt

  val input = filenameToCircuit(inputFile)
  val inputState = CircuitState(input, ChirrtlForm)

  val manager = new TransformManager(factory().prerequisites)
  val preState = manager.execute(inputState)

  hot.util.benchmark(warmup, runs)(factory().transform(preState))
}

object LowerTypesBenchmark extends TransformBenchmark(() => LowerTypes)

object DedupBenchmark extends TransformBenchmark(() => new DedupModules())
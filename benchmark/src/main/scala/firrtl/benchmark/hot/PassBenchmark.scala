// SPDX-License-Identifier: Apache-2.0
package firrtl
package benchmark
package hot

import firrtl.passes._
import firrtl.stage.TransformManager

import firrtl.benchmark.util._

abstract class PassBenchmark(passFactory: () => Pass) extends App {
  val inputFile = args(0)
  val warmup = args(1).toInt
  val runs = args(2).toInt

  val input = filenameToCircuit(inputFile)
  val inputState = CircuitState(input, ChirrtlForm)

  val manager = new TransformManager(passFactory().prerequisites)
  val preState = manager.execute(inputState)

  hot.util.benchmark(warmup, runs)(passFactory().run(preState.circuit))
}

object ResolveKindsBenchmark extends PassBenchmark(() => ResolveKinds)

object CheckHighFormBenchmark extends PassBenchmark(() => CheckHighForm)

object CheckWidthsBenchmark extends PassBenchmark(() => CheckWidths)

object InferBinaryPointsBenchmark extends PassBenchmark(() => new InferBinaryPoints)

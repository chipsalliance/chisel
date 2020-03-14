
package firrtl
package benchmark
package hot

import passes.ResolveKinds
import stage.TransformManager

import firrtl.benchmark.util._

object ResolveKindsBenchmark extends App {
  val inputFile = args(0)
  val warmup = args(1).toInt
  val runs = args(2).toInt

  val input = filenameToCircuit(inputFile)
  val state = CircuitState(input, ChirrtlForm)
  val prereqs = ResolveKinds.prerequisites
  val manager = new TransformManager(prereqs)
  val preState = manager.execute(state)

  hot.util.benchmark(warmup, runs)(ResolveKinds.run(preState.circuit))
}

package chisel3.simulator

import svsim._
import java.lang.ProcessHandle
import chisel3.RawModule

/**
  * `EphemeralSimulator` provides a simple API for ephemeral invocations (such as `scala-cli` scripts) to simulate Chisel modules. To keep things really simple, it only provides the peek/poke API provides enough control while hiding some of the lower-level svsim complexity.
  * The recommended way to use `EphemeralSimulator` is simply to `import chisel3.simulator.EphemeralSimulator._` and then call `simulate(new MyChiselModule()) { module => ... }`.
  */
object EphemeralSimulator extends PeekPokeAPI {

  def simulate[T <: RawModule](
    module: => T
  )(body:   (T) => Unit
  ): Unit = {
    synchronized {
      simulator.simulate(module)({ (_, dut) => body(dut) }).result
    }
  }

  private class DefaultSimulator(val workspacePath: String) extends SingleBackendSimulator[verilator.Backend] {
    val backend = verilator.Backend.initializeFromProcessEnvironment()
    val tag = "default"
    val commonCompilationSettings = CommonCompilationSettings()
    val backendSpecificCompilationSettings = verilator.Backend.CompilationSettings()

    // Try to clean up temporary workspace if possible
    sys.addShutdownHook {
      Runtime.getRuntime().exec(Array("rm", "-rf", workspacePath)).waitFor()
    }
  }
  private lazy val simulator: DefaultSimulator = {
    val temporaryDirectory = System.getProperty("java.io.tmpdir")
    val id = ProcessHandle.current().pid().toString()
    val className = getClass().getName().stripSuffix("$")
    new DefaultSimulator(Seq(temporaryDirectory, className, id).mkString("/"))
  }
}

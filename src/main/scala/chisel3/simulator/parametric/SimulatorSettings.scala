package chisel3.simulator.parametric

import svsim.verilator
import verilator.Backend.CompilationSettings.TraceStyle

/** Trait to represent the simulator settings */
trait SimulatorSettings

protected sealed trait TraceSetting extends SimulatorSettings {
  val traceStyle: TraceStyle
  val extension:  String
}

/**
  * [[TraceVcd]] is an abstracted representation of the
  * [[verilator.Backend.CompilationSettings.TraceStyle.Vcd]] that can be used by
  * the user of the simulator.
  */
private[simulator] case class TraceVcd(traceUnderscore: Boolean) extends TraceSetting with SimulatorSettings {
  val traceStyle: TraceStyle = TraceStyle.Vcd(traceUnderscore)
  val extension:  String = "vcd"
}

/**
  * [[TraceName]] stores the name of the final trace file.
  */
private[simulator] case class TraceName(name: String) extends SimulatorSettings

/**
  * [[SaveWorkspace]] tells to save the workspace of the simulation and stores
  * the final name.
  */
private[simulator] case class SaveWorkspace(name: String) extends SimulatorSettings

/**
  * [[FirtoolArgs]] stores the arguments to pass to the firtool command.
  */
private[simulator] case class FirtoolArgs(args: Seq[String]) extends SimulatorSettings

/**
  * Package object to expose the simulator settings to the user. The following
  * settings can be used by a simulator to allow users to configure the
  * simulation.
  */
package object simulatorSettings {
  // Interface to the simulation
  val VcdTrace:               TraceVcd = TraceVcd(false)
  val VcdTraceWithUnderscore: TraceVcd = TraceVcd(true)

  val SaveWorkdirFile: String => SaveWorkspace = (name: String) => SaveWorkspace(name)
  val SaveWorkdir:     SaveWorkspace = SaveWorkspace("")

  val NameTrace: String => TraceName = (name: String) => TraceName(name)

  val WithFirtoolArgs: Seq[String] => FirtoolArgs = (args: Seq[String]) => FirtoolArgs(args)

}

// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

import java.io.{BufferedWriter, File, FileWriter}
import scala.collection.mutable.ArrayBuffer

/** Represents a single traced module elaboration event. */
private[chisel3] case class TraceEvent(
  val parent:     Option[TraceEvent],
  val startNanos: Long,
  var moduleName: String = "",
  var endNanos:   Long = 0
)

/** Helper to track nested timing information for module elaboration.
  *
  * This class accumulates nested timing information during Chisel elaboration
  * by measuring the time between calls to `pushModule` and `popModule`, which
  * should surround elaboration of a single module.
  *
  * Since this is an internal debugging feature, tracing is not really exposed
  * as an option to the user, but instead is enabled when `CHISEL_TRACE_FILE`
  * environment variable is set to an output path. The resulting trace file can
  * be visualized using [flamegraph] or [inferno]:
  *
  * ```bash
  * # Run Chisel elaboration with environment variable set:
  * CHISEL_TRACE_FILE=$PWD/trace.txt <chisel command>
  * # Visualize using flamegraph.pl:
  * flamegraph.pl trace.txt > trace.svg
  * # Visualize using inferno:
  * inferno-flamegraph trace.txt > trace.svg
  * ```
  *
  * [flamegraph]: https://github.com/brendangregg/FlameGraph
  * [inferno]: https://github.com/jonhoo/inferno
  */
private[chisel3] class ElaborationTrace {
  private val traceFilePath: Option[String] =
    sys.props.get("chisel.trace.file").orElse(sys.env.get("CHISEL_TRACE_FILE"))
  private val enabled: Boolean = traceFilePath.isDefined

  private val events:       ArrayBuffer[TraceEvent] = ArrayBuffer.empty
  private var currentEvent: Option[TraceEvent] = None

  /** Start a new trace event for a module. This should be called before
    * evaluating its constructor. */
  def pushModule(): Unit = if (enabled) {
    currentEvent = Some(TraceEvent(currentEvent, System.nanoTime()))
  }

  /** Stop the tracing for a module. This should be called after evaluating its
    * constructor. Must have a matching earlier `pushModule` call. */
  def popModule(moduleName: String): Unit = if (enabled) {
    val event = currentEvent.get
    event.moduleName = moduleName
    event.endNanos = System.nanoTime()
    events += event
    currentEvent = event.parent
  }

  /** Finish tracing and write results to file if tracing is enabled. */
  def finish(): Unit = traceFilePath.foreach { path =>
    val writer = new BufferedWriter(new FileWriter(new File(path)))
    try {
      for (event <- events) {
        // Build stack by walking parent chain
        var stack = List(event.moduleName)
        var parent = event.parent
        while (parent.isDefined) {
          stack = parent.get.moduleName :: stack
          parent = parent.get.parent
        }
        val durationMicros = (event.endNanos - event.startNanos) / 1000
        writer.write(s"${stack.mkString(";")} $durationMicros\n")
      }
    } finally {
      writer.close()
    }
  }
}

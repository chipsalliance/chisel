package chisel3.libs

import chisel3._
import chisel3.core.{BaseModule, ChiselAnnotation, RunFirrtlTransform, dontTouch}
import chisel3.experimental.{MultiIOModule, RawModule, annotate}
import BreakPoint.BreakPointAnnotation
import firrtl.{AnnotationSeq, CircuitForm, CircuitState, HighForm, LowForm, MidForm, RegKind, RenameMap, Transform, WRef}
import firrtl.annotations._
import firrtl.ir.{Input => _, Module => _, Output => _, _}
import firrtl.passes.memlib.AnalysisUtils
import firrtl.passes.memlib.AnalysisUtils.Connects
import firrtl.passes.wiring.WiringInfo

import scala.collection.JavaConverters._
import scala.collection.mutable


/** Options for me to work on
  *
  * Get executing test for BreakPoint
  * Add transform for AssertDelay
  * Start working on Named
  * Work on pitch to Jonathan
  *
  */
object AssertDelay {
  class AssertDelayTransform extends firrtl.Transform {
    override def inputForm: CircuitForm = LowForm
    override def outputForm: CircuitForm = LowForm

    /** Counts the number of registers from sink to source
      *
      * @param expr Expression to walk
      * @param source Source to find
      * @return For each path from sink to source, count number of register crossings
      */
    def countDelays(expr: Expression, source: String, delaySoFar: Int, connections: Connects): collection.Set[Int] = expr match {
      case WRef(`source`, _, _, _) => Set(delaySoFar)
      case WRef(n, _, _, _) if !connections.contains(n) => Set.empty[Int]
      case WRef(r, _, RegKind, _) => countDelays(connections(r), source, delaySoFar + 1, connections)
      case other =>
        val delays = mutable.HashSet[Int]()

        def onExpr(e: Expression): Expression = {
          val delay = countDelays(e, source, delaySoFar, connections)
          delays ++= delay
          e
        }

        other mapExpr onExpr
        delays
    }

    override def execute(state: CircuitState): CircuitState = {
      val adas = state.annotations.collect{ case a: AssertDelayAnnotation => a}
      val adaMap = adas.groupBy(_.enclosingModule.encapsulatingModule.get)

      val moduleMap = state.circuit.modules.map(m => m.name -> m).toMap
      val errors = mutable.ArrayBuffer[String]()

      // TODO: Assert no CMR's, future work should enable this

      state.circuit.modules.foreach { m =>
        val connections = AnalysisUtils.getConnects(m)
        adaMap.getOrElse(m.name, Nil).map { case AssertDelayAnnotation(source, sink, _, delay) =>
          val delays = countDelays(connections(sink.reference.last.value.toString), source.reference.last.value.toString, 0, connections)
          if(delays.size != 1 || delays.head != delay) {
            errors += s"Delay from ${sink} to ${source} is not $delay, its $delays\n"
          }
        }
      }

      if(errors.nonEmpty) {
        throw new Exception(errors.mkString("\n"))
      }

      state
    }
  }

  case class AssertDelayChiselAnnotation[T<:BaseModule](source: Data, sink: Data, enclosingModule: T, delay: Int) extends RunFirrtlTransform {
    override def toFirrtl: Annotation = AssertDelayAnnotation(source.toNamed, sink.toNamed, enclosingModule.toNamed, delay)
    override def transformClass: Class[_ <: Transform] = classOf[AssertDelay.AssertDelayTransform]
  }

  case class AssertDelayAnnotation(source: Component, sink: Component, enclosingModule: Component, delay: Int) extends Annotation {
    private val errors = mutable.ArrayBuffer[String]()
    private def rename(n: Component, renames: RenameMap): Component = renames.get(n) match {
      case Some(Seq(x: Component)) => x
      case None => n
      case other =>
        errors += s"Bad rename in ${this.getClass}: $n to $other"
        n
    }
    override def update(renames: RenameMap): Seq[Annotation] = {
      val newSource = rename(source, renames)
      val newSink = rename(sink, renames)
      val newEncl = rename(enclosingModule, renames)
      if(errors.nonEmpty) {
        throw new Exception(errors.mkString("\n"))
      }
      Seq(AssertDelayAnnotation(newSource, newSink, newEncl, delay))
    }
  }

  /**
    *
    * @param name Ref of the hardware breakpoint instance
    * @param root Location where the breakpoint will live
    * @param f Function to build breakpoint hardware
    * @tparam T Type of the root hardware
    * @return BreakPoint annotation
    */
  def apply[T<: BaseModule](root: T, source: Data, sink: Data, delay: Int): ChiselAnnotation = {

    // Return Annotations
    AssertDelayChiselAnnotation(source, sink, root, delay)
  }

}

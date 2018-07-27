package chisel3.libs.aspect

import chisel3.experimental.{BaseModule, ChiselAnnotation, MultiIOModule, RawModule}
import chisel3.internal.firrtl.Circuit
import chisel3.{Data, Driver}
import firrtl.ir.{DefInstance, NoInfo}

import scala.util.DynamicVariable


/**
  * Things to think about
  *
  * Who is root
  *
  */
object Aspect {
  import CrossModule.ref2cmr
  private[aspect] val dynamicContextVar = new DynamicVariable[Option[MonitorModule]](None)
  private[aspect] def withAspect[S](m: MonitorModule)(thunk: => S): S = {
    dynamicContextVar.withValue(Some(m))(thunk)
  }
  def getFirrtl(chiselIR: chisel3.internal.firrtl.Circuit): firrtl.ir.DefModule = {
    // Build FIRRTL AST
    val firrtlString = chisel3.internal.firrtl.Emitter.emit(chiselIR)
    val firrtlIR = firrtl.Parser.parse(firrtlString)
    firrtlIR.modules.head
  }
  def getAnnotations(chiselIR: Circuit, dut: Aspect, connections: Seq[(Data, Data)], parent: BaseModule): Seq[ChiselAnnotation] = {

    val firrtlModule = getFirrtl(chiselIR)

    // Return Annotations
    Seq(
      AspectAnnotation(
        connections.map{ case (from, to) => (from.r.getNamed, to.r.getNamed)},
        parent.toNamed,
        DefInstance(NoInfo, dut.instName, firrtlModule.name),
        firrtlModule)
    ) ++ chiselIR.annotations
  }

  def apply[M<: MultiIOModule, T<:Data](instanceName: String,
                                        parent: M,
                                        f: Snippet[M, T]
                                       ): (MonitorModule, Seq[ChiselAnnotation]) = {
    val connections = (parent: M, dut: MonitorModule) => {
      dut.cmrs.toMap ++ Map((parent.clock, dut.clock), (parent.reset, dut.reset))
    }
    apply(instanceName, parent, () => new MonitorModule(instanceName, parent).snip(f), connections)
  }

  def apply[M<: MultiIOModule, S<:Aspect with RawModule](instanceName: String,
                                                         parent: M,
                                                         aspect: () => S,
                                                         connections: (M, S) => Map[Data, Data]
                                                        ): (S, Seq[ChiselAnnotation]) = {
    // Elaborate aspect
    val (chiselIR, dut) = Driver.elaborateAndReturn(aspect)
    val connects = connections(parent, dut)
    (dut, getAnnotations(chiselIR, dut, connects.toSeq, parent))
  }
}

trait Aspect {
  this: RawModule =>
  def instName: String
  def parent: BaseModule
}


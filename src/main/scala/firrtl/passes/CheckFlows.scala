// SPDX-License-Identifier: Apache-2.0

package firrtl.passes

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.traversals.Foreachers._
import firrtl.options.Dependency

object CheckFlows extends Pass {

  override def prerequisites = Dependency(passes.ResolveFlows) +: firrtl.stage.Forms.WorkingIR

  override def optionalPrerequisiteOf =
    Seq(
      Dependency[passes.InferBinaryPoints],
      Dependency[passes.TrimIntervals],
      Dependency[passes.InferWidths],
      Dependency[transforms.InferResets]
    )

  override def invalidates(a: Transform) = false

  type FlowMap = collection.mutable.HashMap[String, Flow]

  implicit def toStr(g: Flow): String = g match {
    case SourceFlow  => "source"
    case SinkFlow    => "sink"
    case UnknownFlow => "unknown"
    case DuplexFlow  => "duplex"
  }

  class WrongFlow(info: Info, mname: String, expr: String, wrong: Flow, right: Flow)
      extends PassException(
        s"$info: [module $mname]  Expression $expr is used as a $wrong but can only be used as a $right."
      )

  def run(c: Circuit): Circuit = {
    val errors = new Errors()

    def get_flow(e: Expression, flows: FlowMap): Flow = e match {
      case (e: WRef)      => flows(e.name)
      case (e: WSubIndex) => get_flow(e.expr, flows)
      case (e: WSubAccess) => get_flow(e.expr, flows)
      case (e: WSubField) =>
        e.expr.tpe match {
          case t: BundleType =>
            val f = (t.fields.find(_.name == e.name)).get
            times(get_flow(e.expr, flows), f.flip)
        }
      case _ => SourceFlow
    }

    def flip_q(t: Type): Boolean = {
      def flip_rec(t: Type, f: Orientation): Boolean = t match {
        case tx: BundleType => tx.fields.exists(field => flip_rec(field.tpe, times(f, field.flip)))
        case tx: VectorType => flip_rec(tx.tpe, f)
        case tx => f == Flip
      }
      flip_rec(t, Default)
    }

    def check_flow(info: Info, mname: String, flows: FlowMap, desired: Flow)(e: Expression): Unit = {
      val flow = get_flow(e, flows)
      (flow, desired) match {
        case (SourceFlow, SinkFlow) =>
          errors.append(new WrongFlow(info, mname, e.serialize, desired, flow))
        case (SinkFlow, SourceFlow) =>
          kind(e) match {
            case PortKind | InstanceKind if !flip_q(e.tpe) => // OK!
            case _ =>
              errors.append(new WrongFlow(info, mname, e.serialize, desired, flow))
          }
        case _ =>
      }
    }

    def check_flows_e(info: Info, mname: String, flows: FlowMap)(e: Expression): Unit = {
      e match {
        case e: Mux    => e.foreach(check_flow(info, mname, flows, SourceFlow))
        case e: DoPrim => e.args.foreach(check_flow(info, mname, flows, SourceFlow))
        case _ =>
      }
      e.foreach(check_flows_e(info, mname, flows))
    }

    def check_flows_s(minfo: Info, mname: String, flows: FlowMap)(s: Statement): Unit = {
      val info = get_info(s) match {
        case NoInfo => minfo
        case x      => x
      }
      s match {
        case (s: DefWire)     => flows(s.name) = DuplexFlow
        case (s: DefRegister) => flows(s.name) = DuplexFlow
        case (s: DefMemory)   => flows(s.name) = SourceFlow
        case (s: WDefInstance) => flows(s.name) = SourceFlow
        case (s: DefNode) =>
          check_flow(info, mname, flows, SourceFlow)(s.value)
          flows(s.name) = SourceFlow
        case (s: Connect) =>
          check_flow(info, mname, flows, SinkFlow)(s.loc)
          check_flow(info, mname, flows, SourceFlow)(s.expr)
        case (s: Print) =>
          s.args.foreach(check_flow(info, mname, flows, SourceFlow))
          check_flow(info, mname, flows, SourceFlow)(s.en)
          check_flow(info, mname, flows, SourceFlow)(s.clk)
        case (s: PartialConnect) =>
          check_flow(info, mname, flows, SinkFlow)(s.loc)
          check_flow(info, mname, flows, SourceFlow)(s.expr)
        case (s: Conditionally) =>
          check_flow(info, mname, flows, SourceFlow)(s.pred)
        case (s: Stop) =>
          check_flow(info, mname, flows, SourceFlow)(s.en)
          check_flow(info, mname, flows, SourceFlow)(s.clk)
        case (s: Verification) =>
          check_flow(info, mname, flows, SourceFlow)(s.clk)
          check_flow(info, mname, flows, SourceFlow)(s.pred)
          check_flow(info, mname, flows, SourceFlow)(s.en)
        case _ =>
      }
      s.foreach(check_flows_e(info, mname, flows))
      s.foreach(check_flows_s(minfo, mname, flows))
    }

    for (m <- c.modules) {
      val flows = new FlowMap
      flows ++= (m.ports.map(p => p.name -> to_flow(p.direction)))
      m.foreach(check_flows_s(m.info, m.name, flows))
    }
    errors.trigger()
    c
  }
}

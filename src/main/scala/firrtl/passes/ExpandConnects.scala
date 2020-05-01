package firrtl.passes

import firrtl.Utils.{create_exps, flow, get_field, get_valid_points, times, to_flip, to_flow}
import firrtl.ir._
import firrtl.options.Dependency
import firrtl.{DuplexFlow, Flow, SinkFlow, SourceFlow, Transform, WDefInstance, WRef, WSubAccess, WSubField, WSubIndex}
import firrtl.Mappers._

object ExpandConnects extends Pass {

  override def prerequisites =
    Seq( Dependency(PullMuxes),
         Dependency(ReplaceAccesses) ) ++ firrtl.stage.Forms.Deduped

  override def invalidates(a: Transform) = a match {
    case ResolveFlows => true
    case _            => false
  }

  def run(c: Circuit): Circuit = {
    def expand_connects(m: Module): Module = {
      val flows = collection.mutable.LinkedHashMap[String,Flow]()
      def expand_s(s: Statement): Statement = {
        def set_flow(e: Expression): Expression = e map set_flow match {
          case ex: WRef => WRef(ex.name, ex.tpe, ex.kind, flows(ex.name))
          case ex: WSubField =>
            val f = get_field(ex.expr.tpe, ex.name)
            val flowx = times(flow(ex.expr), f.flip)
            WSubField(ex.expr, ex.name, ex.tpe, flowx)
          case ex: WSubIndex => WSubIndex(ex.expr, ex.value, ex.tpe, flow(ex.expr))
          case ex: WSubAccess => WSubAccess(ex.expr, ex.index, ex.tpe, flow(ex.expr))
          case ex => ex
        }
        s match {
          case sx: DefWire => flows(sx.name) = DuplexFlow; sx
          case sx: DefRegister => flows(sx.name) = DuplexFlow; sx
          case sx: WDefInstance => flows(sx.name) = SourceFlow; sx
          case sx: DefMemory => flows(sx.name) = SourceFlow; sx
          case sx: DefNode => flows(sx.name) = SourceFlow; sx
          case sx: IsInvalid =>
            val invalids = create_exps(sx.expr).flatMap { case expx =>
               flow(set_flow(expx)) match {
                  case DuplexFlow => Some(IsInvalid(sx.info, expx))
                  case SinkFlow => Some(IsInvalid(sx.info, expx))
                  case _ => None
               }
            }
            invalids.size match {
               case 0 => EmptyStmt
               case 1 => invalids.head
               case _ => Block(invalids)
            }
          case sx: Connect =>
            val locs = create_exps(sx.loc)
            val exps = create_exps(sx.expr)
            Block(locs.zip(exps).map { case (locx, expx) =>
               to_flip(flow(locx)) match {
                  case Default => Connect(sx.info, locx, expx)
                  case Flip => Connect(sx.info, expx, locx)
               }
            })
          case sx: PartialConnect =>
            val ls = get_valid_points(sx.loc.tpe, sx.expr.tpe, Default, Default)
            val locs = create_exps(sx.loc)
            val exps = create_exps(sx.expr)
            val stmts = ls map { case (x, y) =>
              locs(x).tpe match {
                case AnalogType(_) => Attach(sx.info, Seq(locs(x), exps(y)))
                case _ =>
                  to_flip(flow(locs(x))) match {
                    case Default => Connect(sx.info, locs(x), exps(y))
                    case Flip => Connect(sx.info, exps(y), locs(x))
                  }
              }
            }
            Block(stmts)
          case sx => sx map expand_s
        }
      }

      m.ports.foreach { p => flows(p.name) = to_flow(p.direction) }
      Module(m.info, m.name, m.ports, expand_s(m.body))
    }

    val modulesx = c.modules.map {
       case (m: ExtModule) => m
       case (m: Module) => expand_connects(m)
    }
    Circuit(c.info, modulesx, c.main)
  }
}

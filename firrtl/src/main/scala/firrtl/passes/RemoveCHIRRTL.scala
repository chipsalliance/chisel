// SPDX-License-Identifier: Apache-2.0

package firrtl.passes

// Datastructures
import scala.collection.mutable.ArrayBuffer
import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import firrtl.options.Dependency
import firrtl.renamemap.MutableRenameMap

case class MPort(name: String, clk: Expression)
case class MPorts(readers: ArrayBuffer[MPort], writers: ArrayBuffer[MPort], readwriters: ArrayBuffer[MPort])
case class DataRef(exp: Expression, source: String, sink: String, mask: String, rdwrite: Boolean)

object RemoveCHIRRTL extends Transform with DependencyAPIMigration {

  override def prerequisites = firrtl.stage.Forms.ChirrtlForm

  override def invalidates(a: Transform) = false

  val ut = UnknownType
  type MPortMap = collection.mutable.LinkedHashMap[String, MPorts]
  type SeqMemSet = collection.mutable.HashSet[String]
  type MPortTypeMap = collection.mutable.LinkedHashMap[String, Type]
  type DataRefMap = collection.mutable.LinkedHashMap[String, DataRef]
  type AddrMap = collection.mutable.HashMap[String, Expression]

  def create_all_exps(ex: Expression): Seq[Expression] = ex.tpe match {
    case _: GroundType => Seq(ex)
    case t: BundleType =>
      (t.fields.foldLeft(Seq[Expression]()))((exps, f) => exps ++ create_all_exps(SubField(ex, f.name, f.tpe))) ++ Seq(
        ex
      )
    case t: VectorType =>
      ((0 until t.size).foldLeft(Seq[Expression]()))((exps, i) =>
        exps ++ create_all_exps(SubIndex(ex, i, t.tpe))
      ) ++ Seq(ex)
    case UnknownType => Seq(ex)
  }

  def create_exps(e: Expression): Seq[Expression] = e match {
    case ex: Mux =>
      val e1s = create_exps(ex.tval)
      val e2s = create_exps(ex.fval)
      (e1s.zip(e2s)).map { case (e1, e2) => Mux(ex.cond, e1, e2, mux_type(e1, e2)) }
    case ex: ValidIf =>
      create_exps(ex.value).map(e1 => ValidIf(ex.cond, e1, e1.tpe))
    case ex =>
      ex.tpe match {
        case _: GroundType => Seq(ex)
        case t: BundleType =>
          (t.fields.foldLeft(Seq[Expression]()))((exps, f) => exps ++ create_exps(SubField(ex, f.name, f.tpe)))
        case t: VectorType =>
          ((0 until t.size).foldLeft(Seq[Expression]()))((exps, i) => exps ++ create_exps(SubIndex(ex, i, t.tpe)))
        case UnknownType => Seq(ex)
      }
  }

  private def EMPs: MPorts = MPorts(ArrayBuffer[MPort](), ArrayBuffer[MPort](), ArrayBuffer[MPort]())

  def collect_smems_and_mports(mports: MPortMap, smems: SeqMemSet)(s: Statement): Statement = {
    s match {
      case sx: CDefMemory if sx.seq => smems += sx.name
      case sx: CDefMPort =>
        val p = mports.getOrElse(sx.mem, EMPs)
        sx.direction match {
          case MRead      => p.readers += MPort(sx.name, sx.exps(1))
          case MWrite     => p.writers += MPort(sx.name, sx.exps(1))
          case MReadWrite => p.readwriters += MPort(sx.name, sx.exps(1))
          case MInfer     => // direction may not be inferred if it's not being used
        }
        mports(sx.mem) = p
      case _ =>
    }
    s.map(collect_smems_and_mports(mports, smems))
  }

  @deprecated("This should never have been public", "FIRRTL 1.5")
  def collect_refs(
    mports:  MPortMap,
    smems:   SeqMemSet,
    types:   MPortTypeMap,
    refs:    DataRefMap,
    raddrs:  AddrMap,
    renames: RenameMap
  )(s:       Statement
  ): Statement =
    // Cast is safe because RenameMap is sealed trait, MutableRenameMap is only subclass
    collect_refs(mports, smems, types, refs, raddrs, renames.asInstanceOf[MutableRenameMap])(s)

  private def collect_refs(
    mports:  MPortMap,
    smems:   SeqMemSet,
    types:   MPortTypeMap,
    refs:    DataRefMap,
    raddrs:  AddrMap,
    renames: MutableRenameMap
  )(s:       Statement
  ): Statement = s match {
    case sx: CDefMemory =>
      types(sx.name) = sx.tpe
      val taddr = UIntType(IntWidth(1.max(getUIntWidth(sx.size - 1))))
      val tdata = sx.tpe
      def set_poison(vec: scala.collection.Seq[MPort]) = vec.toSeq.flatMap(r =>
        Seq(
          IsInvalid(sx.info, SubField(SubField(Reference(sx.name, ut), r.name, ut), "addr", taddr)),
          IsInvalid(sx.info, SubField(SubField(Reference(sx.name, ut), r.name, ut), "clk", ClockType))
        )
      )
      def set_enable(vec: scala.collection.Seq[MPort], en: String) =
        vec.toSeq.map(r => Connect(sx.info, SubField(SubField(Reference(sx.name, ut), r.name, ut), en, BoolType), zero))
      def set_write(vec: scala.collection.Seq[MPort], data: String, mask: String) = vec.toSeq.flatMap { r =>
        val tmask = createMask(sx.tpe)
        val portRef = SubField(Reference(sx.name, ut), r.name, ut)
        Seq(IsInvalid(sx.info, SubField(portRef, data, tdata)), IsInvalid(sx.info, SubField(portRef, mask, tmask)))
      }
      val rds = (mports.getOrElse(sx.name, EMPs)).readers
      val wrs = (mports.getOrElse(sx.name, EMPs)).writers
      val rws = (mports.getOrElse(sx.name, EMPs)).readwriters
      val stmts = set_poison(rds) ++
        set_enable(rds, "en") ++
        set_poison(wrs) ++
        set_enable(wrs, "en") ++
        set_write(wrs, "data", "mask") ++
        set_poison(rws) ++
        set_enable(rws, "wmode") ++
        set_enable(rws, "en") ++
        set_write(rws, "wdata", "wmask")
      val mem = DefMemory(
        sx.info,
        sx.name,
        sx.tpe,
        sx.size,
        1,
        if (sx.seq) 1 else 0,
        rds.map(_.name).toSeq,
        wrs.map(_.name).toSeq,
        rws.map(_.name).toSeq,
        sx.readUnderWrite
      )
      Block(mem +: stmts)
    case sx: CDefMPort =>
      types.get(sx.mem) match {
        case Some(mem) => types(sx.name) = mem
        case None =>
          throw new PassException(s"Undefined memory ${sx.mem} referenced by mport ${sx.name}")
      }
      val addrs = ArrayBuffer[String]()
      val clks = ArrayBuffer[String]()
      val ens = ArrayBuffer[String]()
      val masks = ArrayBuffer.empty[Expression]
      val portRef = SubField(Reference(sx.mem, ut), sx.name, ut)
      sx.direction match {
        case MReadWrite =>
          refs(sx.name) = DataRef(portRef, "rdata", "wdata", "wmask", rdwrite = true)
          addrs += "addr"
          clks += "clk"
          ens += "en"
          masks ++= create_exps(SubField(portRef, "wmask", createMask(sx.tpe)))
          renames.rename(sx.name, s"${sx.mem}.${sx.name}.rdata")
          renames.rename(sx.name, s"${sx.mem}.${sx.name}.wdata")
          val es = create_all_exps(WRef(sx.name, sx.tpe))
          val rs = create_all_exps(WRef(s"${sx.mem}.${sx.name}.rdata", sx.tpe))
          val ws = create_all_exps(WRef(s"${sx.mem}.${sx.name}.wdata", sx.tpe))
          ((es.zip(rs)).zip(ws)).map {
            case ((e, r), w) => renames.rename(e.serialize, Seq(r.serialize, w.serialize))
          }
        case MWrite =>
          refs(sx.name) = DataRef(portRef, "data", "data", "mask", rdwrite = false)
          addrs += "addr"
          clks += "clk"
          ens += "en"
          masks ++= create_exps(SubField(portRef, "mask", createMask(sx.tpe)))
          renames.rename(sx.name, s"${sx.mem}.${sx.name}.data")
          val es = create_all_exps(WRef(sx.name, sx.tpe))
          val ws = create_all_exps(WRef(s"${sx.mem}.${sx.name}.data", sx.tpe))
          (es.zip(ws)).map {
            case (e, w) => renames.rename(e.serialize, w.serialize)
          }
        case MRead =>
          refs(sx.name) = DataRef(portRef, "data", "data", "blah", rdwrite = false)
          addrs += "addr"
          clks += "clk"
          sx.exps.head match {
            case e: Reference if smems(sx.mem) =>
              raddrs(e.name) = SubField(portRef, "en", ut)
            case _ => ens += "en"
          }
          renames.rename(sx.name, s"${sx.mem}.${sx.name}.data")
          val es = create_all_exps(WRef(sx.name, sx.tpe))
          val rs = create_all_exps(WRef(s"${sx.mem}.${sx.name}.data", sx.tpe))
          (es.zip(rs)).map {
            case (e, r) => renames.rename(e.serialize, r.serialize)
          }
        case MInfer => // do nothing if it's not being used
      }
      Block(
        List() ++
          (addrs.map(x => Connect(sx.info, SubField(portRef, x, ut), sx.exps.head))) ++
          (clks.map(x => Connect(sx.info, SubField(portRef, x, ut), sx.exps(1)))) ++
          (ens.map(x => Connect(sx.info, SubField(portRef, x, ut), one))) ++
          masks.map(lhs => Connect(sx.info, lhs, zero))
      )
    case sx => sx.map(collect_refs(mports, smems, types, refs, raddrs, renames))
  }

  def get_mask(refs: DataRefMap)(e: Expression): Expression =
    e.map(get_mask(refs)) match {
      case ex: Reference =>
        refs.get(ex.name) match {
          case None    => ex
          case Some(p) => SubField(p.exp, p.mask, createMask(ex.tpe))
        }
      case ex => ex
    }

  def remove_chirrtl_s(refs: DataRefMap, raddrs: AddrMap)(s: Statement): Statement = {
    var has_write_mport = false
    var has_readwrite_mport: Option[Expression] = None
    var has_read_mport:      Option[Expression] = None
    def remove_chirrtl_e(g: Flow)(e: Expression): Expression = e match {
      case Reference(name, tpe, _, _) =>
        refs.get(name) match {
          case Some(p) =>
            g match {
              case SinkFlow =>
                has_write_mport = true
                if (p.rdwrite) has_readwrite_mport = Some(SubField(p.exp, "wmode", BoolType))
                SubField(p.exp, p.sink, tpe)
              case SourceFlow =>
                SubField(p.exp, p.source, tpe)
            }
          case None =>
            g match {
              case SinkFlow =>
                raddrs.get(name) match {
                  case Some(en) => has_read_mport = Some(en); e
                  case None     => e
                }
              case SourceFlow => e
            }
        }
      case SubAccess(expr, index, tpe, _) =>
        SubAccess(remove_chirrtl_e(g)(expr), remove_chirrtl_e(SourceFlow)(index), tpe)
      case ex => ex.map(remove_chirrtl_e(g))
    }
    s match {
      case DefNode(info, name, value) =>
        val valuex = remove_chirrtl_e(SourceFlow)(value)
        val sx = DefNode(info, name, valuex)
        // Check node is used for read port address
        remove_chirrtl_e(SinkFlow)(Reference(name, value.tpe))
        has_read_mport match {
          case None     => sx
          case Some(en) => Block(sx, Connect(info, en, one))
        }
      case Connect(info, loc, expr) =>
        val rocx = remove_chirrtl_e(SourceFlow)(expr)
        val locx = remove_chirrtl_e(SinkFlow)(loc)
        val sx = Connect(info, locx, rocx)
        val stmts = ArrayBuffer[Statement]()
        has_read_mport match {
          case None     =>
          case Some(en) => stmts += Connect(info, en, one)
        }
        if (has_write_mport) {
          val locs = create_exps(get_mask(refs)(loc))
          stmts ++= (locs.map(x => Connect(info, x, one)))
          has_readwrite_mport match {
            case None        =>
            case Some(wmode) => stmts += Connect(info, wmode, one)
          }
        }
        if (stmts.isEmpty) sx else Block(sx +: stmts.toSeq)
      case PartialConnect(info, loc, expr) =>
        val locx = remove_chirrtl_e(SinkFlow)(loc)
        val rocx = remove_chirrtl_e(SourceFlow)(expr)
        val sx = PartialConnect(info, locx, rocx)
        val stmts = ArrayBuffer[Statement]()
        has_read_mport match {
          case None     =>
          case Some(en) => stmts += Connect(info, en, one)
        }
        if (has_write_mport) {
          val ls = get_valid_points(loc.tpe, expr.tpe, Default, Default)
          val locs = create_exps(get_mask(refs)(loc))
          stmts ++= (ls.map { case (x, _) => Connect(info, locs(x), one) })
          has_readwrite_mport match {
            case None        =>
            case Some(wmode) => stmts += Connect(info, wmode, one)
          }
        }
        if (stmts.isEmpty) sx else Block(sx +: stmts.toSeq)
      case sx => sx.map(remove_chirrtl_s(refs, raddrs)).map(remove_chirrtl_e(SourceFlow))
    }
  }

  @deprecated("This should never have been public", "FIRRTL 1.5")
  def remove_chirrtl_m(renames: RenameMap)(m: DefModule): DefModule =
    // Cast is safe because RenameMap is sealed trait, MutableRenameMap is only subclass
    remove_chirrtl_m(renames.asInstanceOf[MutableRenameMap])(m)

  private def remove_chirrtl_m(renames: MutableRenameMap)(m: DefModule): DefModule = {
    val mports = new MPortMap
    val smems = new SeqMemSet
    val types = new MPortTypeMap
    val refs = new DataRefMap
    val raddrs = new AddrMap
    renames.setModule(m.name)
    (m.map(collect_smems_and_mports(mports, smems))
      .map(collect_refs(mports, smems, types, refs, raddrs, renames))
      .map(remove_chirrtl_s(refs, raddrs)))
  }

  def execute(state: CircuitState): CircuitState = {
    val c = state.circuit
    val renames = MutableRenameMap()
    renames.setCircuit(c.main)
    val result = c.copy(modules = c.modules.map(remove_chirrtl_m(renames)))
    state.copy(circuit = result, renames = Some(renames))
  }
}

// See LICENSE for license details.

package firrtl.passes

// Datastructures
import scala.collection.mutable.ArrayBuffer

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._

case class MPort(name: String, clk: Expression)
case class MPorts(readers: ArrayBuffer[MPort], writers: ArrayBuffer[MPort], readwriters: ArrayBuffer[MPort])
case class DataRef(exp: Expression, male: String, female: String, mask: String, rdwrite: Boolean)

object RemoveCHIRRTL extends Pass {
  def name = "Remove CHIRRTL"

  val ut = UnknownType
  type MPortMap = collection.mutable.LinkedHashMap[String, MPorts]
  type SeqMemSet = collection.mutable.HashSet[String]
  type MPortTypeMap = collection.mutable.LinkedHashMap[String, Type]
  type DataRefMap = collection.mutable.LinkedHashMap[String, DataRef]
  type AddrMap = collection.mutable.HashMap[String, Expression]

  def create_exps(e: Expression): Seq[Expression] = e match {
    case ex: Mux =>
      val e1s = create_exps(ex.tval)
      val e2s = create_exps(ex.fval)
      (e1s zip e2s) map { case (e1, e2) => Mux(ex.cond, e1, e2, mux_type(e1, e2)) }
    case ex: ValidIf =>
      create_exps(ex.value) map (e1 => ValidIf(ex.cond, e1, e1.tpe))
    case ex => ex.tpe match {
      case _: GroundType => Seq(ex)
      case t: BundleType => (t.fields foldLeft Seq[Expression]())((exps, f) =>
        exps ++ create_exps(SubField(ex, f.name, f.tpe)))
      case t: VectorType => ((0 until t.size) foldLeft Seq[Expression]())((exps, i) =>
        exps ++ create_exps(SubIndex(ex, i, t.tpe)))
      case UnknownType => Seq(ex)
    }
  }

  private def EMPs: MPorts = MPorts(ArrayBuffer[MPort](), ArrayBuffer[MPort](), ArrayBuffer[MPort]())

  def collect_smems_and_mports(mports: MPortMap, smems: SeqMemSet)(s: Statement): Statement = {
    s match {
      case sx: CDefMemory if sx.seq => smems += sx.name
      case sx: CDefMPort =>
        val p = mports getOrElse (sx.mem, EMPs)
        sx.direction match {
          case MRead => p.readers += MPort(sx.name, sx.exps(1))
          case MWrite => p.writers += MPort(sx.name, sx.exps(1))
          case MReadWrite => p.readwriters += MPort(sx.name, sx.exps(1))
          case MInfer => // direction may not be inferred if it's not being used
        }
        mports(sx.mem) = p
      case _ =>
    }
    s map collect_smems_and_mports(mports, smems)
  }

  def collect_refs(mports: MPortMap, smems: SeqMemSet, types: MPortTypeMap,
      refs: DataRefMap, raddrs: AddrMap)(s: Statement): Statement = s match {
    case sx: CDefMemory =>
      types(sx.name) = sx.tpe
      val taddr = UIntType(IntWidth(1 max ceilLog2(sx.size)))
      val tdata = sx.tpe
      def set_poison(vec: Seq[MPort]) = vec flatMap (r => Seq(
        IsInvalid(sx.info, SubField(SubField(Reference(sx.name, ut), r.name, ut), "addr", taddr)),
        IsInvalid(sx.info, SubField(SubField(Reference(sx.name, ut), r.name, ut), "clk", ClockType))
      ))
      def set_enable(vec: Seq[MPort], en: String) = vec map (r =>
        Connect(sx.info, SubField(SubField(Reference(sx.name, ut), r.name, ut), en, BoolType), zero)
      )
      def set_write(vec: Seq[MPort], data: String, mask: String) = vec flatMap {r =>
        val tmask = createMask(sx.tpe)
        IsInvalid(sx.info, SubField(SubField(Reference(sx.name, ut), r.name, ut), data, tdata)) +:
             (create_exps(SubField(SubField(Reference(sx.name, ut), r.name, ut), mask, tmask))
               map (Connect(sx.info, _, zero))
             )
      }
      val rds = (mports getOrElse (sx.name, EMPs)).readers
      val wrs = (mports getOrElse (sx.name, EMPs)).writers
      val rws = (mports getOrElse (sx.name, EMPs)).readwriters
      val stmts = set_poison(rds) ++
        set_enable(rds, "en") ++
        set_poison(wrs) ++
        set_enable(wrs, "en") ++
        set_write(wrs, "data", "mask") ++
        set_poison(rws) ++
        set_enable(rws, "wmode") ++
        set_enable(rws, "en") ++
        set_write(rws, "wdata", "wmask")
      val mem = DefMemory(sx.info, sx.name, sx.tpe, sx.size, 1, if (sx.seq) 1 else 0,
                  rds map (_.name), wrs map (_.name), rws map (_.name))
      Block(mem +: stmts)
    case sx: CDefMPort =>
      types(sx.name) = types(sx.mem)
      val addrs = ArrayBuffer[String]()
      val clks = ArrayBuffer[String]()
      val ens = ArrayBuffer[String]()
      sx.direction match {
        case MReadWrite =>
          refs(sx.name) = DataRef(SubField(Reference(sx.mem, ut), sx.name, ut), "rdata", "wdata", "wmask", rdwrite = true)
          addrs += "addr"
          clks += "clk"
          ens += "en"
        case MWrite =>
          refs(sx.name) = DataRef(SubField(Reference(sx.mem, ut), sx.name, ut), "data", "data", "mask", rdwrite = false)
          addrs += "addr"
          clks += "clk"
          ens += "en"
        case MRead =>
          refs(sx.name) = DataRef(SubField(Reference(sx.mem, ut), sx.name, ut), "data", "data", "blah", rdwrite = false)
          addrs += "addr"
          clks += "clk"
          sx.exps.head match {
            case e: Reference if smems(sx.mem) =>
              raddrs(e.name) = SubField(SubField(Reference(sx.mem, ut), sx.name, ut), "en", ut)
            case _ => ens += "en"
          }
        case MInfer => // do nothing if it's not being used
      }
      Block(
        (addrs map (x => Connect(sx.info, SubField(SubField(Reference(sx.mem, ut), sx.name, ut), x, ut), sx.exps.head))) ++
        (clks map (x => Connect(sx.info, SubField(SubField(Reference(sx.mem, ut), sx.name, ut), x, ut), sx.exps(1)))) ++
        (ens map (x => Connect(sx.info,SubField(SubField(Reference(sx.mem,ut), sx.name, ut), x, ut), one))))
    case sx => sx map collect_refs(mports, smems, types, refs, raddrs)
  }

  def get_mask(refs: DataRefMap)(e: Expression): Expression =
    e map get_mask(refs) match {
      case ex: Reference => refs get ex.name match {
        case None => ex
        case Some(p) => SubField(p.exp, p.mask, createMask(ex.tpe))
      }
      case ex => ex
    }

  def remove_chirrtl_s(refs: DataRefMap, raddrs: AddrMap)(s: Statement): Statement = {
    var has_write_mport = false
    var has_readwrite_mport: Option[Expression] = None
    var has_read_mport: Option[Expression] = None
    def remove_chirrtl_e(g: Gender)(e: Expression): Expression = e match {
      case Reference(name, tpe) => refs get name match {
        case Some(p) => g match {
          case FEMALE =>
            has_write_mport = true
            if (p.rdwrite) has_readwrite_mport = Some(SubField(p.exp, "wmode", BoolType))
            SubField(p.exp, p.female, tpe)
          case MALE =>
            SubField(p.exp, p.male, tpe)
        }
        case None => g match {
          case FEMALE => raddrs get name match {
            case Some(en) => has_read_mport = Some(en) ; e
            case None => e
          }
          case MALE => e
        }
      }
      case SubAccess(expr, index, tpe)  => SubAccess(
        remove_chirrtl_e(g)(expr), remove_chirrtl_e(MALE)(index), tpe)
      case ex => ex map remove_chirrtl_e(g)
   }
   s match {
      case DefNode(info, name, value) =>
        val valuex = remove_chirrtl_e(MALE)(value)
        val sx = DefNode(info, name, valuex)
        // Check node is used for read port address
        remove_chirrtl_e(FEMALE)(Reference(name, value.tpe))
        has_read_mport match {
          case None => sx
          case Some(en) => Block(Seq(sx, Connect(info, en, one)))
        }
      case Connect(info, loc, expr) =>
        val rocx = remove_chirrtl_e(MALE)(expr)
        val locx = remove_chirrtl_e(FEMALE)(loc)
        val sx = Connect(info, locx, rocx)
        val stmts = ArrayBuffer[Statement]()
        has_read_mport match {
          case None =>
          case Some(en) => stmts += Connect(info, en, one)
        }
        if (has_write_mport) {
          val locs = create_exps(get_mask(refs)(loc))
          stmts ++= (locs map (x => Connect(info, x, one)))
          has_readwrite_mport match {
            case None =>
            case Some(wmode) => stmts += Connect(info, wmode, one)
          }
        }
        if (stmts.isEmpty) sx else Block(sx +: stmts)
      case PartialConnect(info, loc, expr) =>
        val locx = remove_chirrtl_e(FEMALE)(loc)
        val rocx = remove_chirrtl_e(MALE)(expr)
        val sx = PartialConnect(info, locx, rocx)
        val stmts = ArrayBuffer[Statement]()
        has_read_mport match {
          case None =>
          case Some(en) => stmts += Connect(info, en, one)
        }
        if (has_write_mport) {
          val ls = get_valid_points(loc.tpe, expr.tpe, Default, Default)
          val locs = create_exps(get_mask(refs)(loc))
          stmts ++= (ls map { case (x, _) => Connect(info, locs(x), one) })
          has_readwrite_mport match {
            case None =>
            case Some(wmode) => stmts += Connect(info, wmode, one)
          }
        }
        if (stmts.isEmpty) sx else Block(sx +: stmts)
      case sx => sx map remove_chirrtl_s(refs, raddrs) map remove_chirrtl_e(MALE)
    }
  }

  def remove_chirrtl_m(m: DefModule): DefModule = {
    val mports = new MPortMap
    val smems = new SeqMemSet
    val types = new MPortTypeMap
    val refs = new DataRefMap
    val raddrs = new AddrMap
    (m map collect_smems_and_mports(mports, smems)
       map collect_refs(mports, smems, types, refs, raddrs)
       map remove_chirrtl_s(refs, raddrs))
  }

  def run(c: Circuit): Circuit =
    c copy (modules = c.modules map remove_chirrtl_m)
}

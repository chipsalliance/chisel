/*
Copyright (c) 2014 - 2016 The Regents of the University of
California (Regents). All Rights Reserved.  Redistribution and use in
source and binary forms, with or without modification, are permitted
provided that the following conditions are met:
   * Redistributions of source code must retain the above
     copyright notice, this list of conditions and the following
     two paragraphs of disclaimer.
   * Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     two paragraphs of disclaimer in the documentation and/or other materials
     provided with the distribution.
   * Neither the name of the Regents nor the names of its contributors
     may be used to endorse or promote products derived from this
     software without specific prior written permission.
IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF
REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF
ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION
TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
MODIFICATIONS.
*/

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
    case (e: Mux) =>
      val e1s = create_exps(e.tval)
      val e2s = create_exps(e.fval)
      (e1s zip e2s) map { case (e1, e2) => Mux(e.cond, e1, e2, mux_type(e1, e2)) }
    case (e: ValidIf) =>
      create_exps(e.value) map (e1 => ValidIf(e.cond, e1, e1.tpe))
    case (e) => (e.tpe) match {
      case (_: GroundType) => Seq(e)
      case (t: BundleType) => (t.fields foldLeft Seq[Expression]())((exps, f) =>
        exps ++ create_exps(SubField(e, f.name, f.tpe)))
      case (t: VectorType) => ((0 until t.size) foldLeft Seq[Expression]())((exps, i) =>
        exps ++ create_exps(SubIndex(e, i, t.tpe)))
      case UnknownType => Seq(e)
    }
  }

  private def EMPs: MPorts = MPorts(ArrayBuffer[MPort](), ArrayBuffer[MPort](), ArrayBuffer[MPort]())

  def collect_smems_and_mports(mports: MPortMap, smems: SeqMemSet)(s: Statement): Statement = {
    s match {
      case (s:CDefMemory) if s.seq => smems += s.name
      case (s:CDefMPort) =>
        val p = mports getOrElse (s.mem, EMPs)
        s.direction match {
          case MRead => p.readers += MPort(s.name,s.exps(1))
          case MWrite => p.writers += MPort(s.name,s.exps(1))
          case MReadWrite => p.readwriters += MPort(s.name,s.exps(1))
        }
        mports(s.mem) = p
      case s =>
    }
    s map collect_smems_and_mports(mports, smems)
  }

  def collect_refs(mports: MPortMap, smems: SeqMemSet, types: MPortTypeMap,
      refs: DataRefMap, raddrs: AddrMap)(s: Statement): Statement = s match {
    case (s: CDefMemory) =>
      types(s.name) = s.tpe
      val taddr = UIntType(IntWidth(math.max(1, ceil_log2(s.size))))
      val tdata = s.tpe
      def set_poison(vec: Seq[MPort], addr: String) = vec flatMap (r => Seq(
        IsInvalid(s.info, SubField(SubField(Reference(s.name, ut), r.name, ut), addr, taddr)),
        IsInvalid(s.info, SubField(SubField(Reference(s.name, ut), r.name, ut), "clk", taddr))
      ))
      def set_enable(vec: Seq[MPort], en: String) = vec map (r =>
        Connect(s.info, SubField(SubField(Reference(s.name, ut), r.name, ut), en, taddr), zero)
      )
      def set_wmode (vec: Seq[MPort], wmode: String) = vec map (r =>
        Connect(s.info, SubField(SubField(Reference(s.name, ut), r.name, ut), wmode, taddr), zero)
      )
      def set_write (vec: Seq[MPort], data: String, mask: String) = vec flatMap {r =>
        val tmask = createMask(s.tpe)
        IsInvalid(s.info, SubField(SubField(Reference(s.name, ut), r.name, ut), data, tdata)) +:
             (create_exps(SubField(SubField(Reference(s.name, ut), r.name, ut), mask, tmask))
               map (Connect(s.info, _, zero))
             )
      }
      val rds = (mports getOrElse (s.name, EMPs)).readers
      val wrs = (mports getOrElse (s.name, EMPs)).writers
      val rws = (mports getOrElse (s.name, EMPs)).readwriters
      val stmts = set_poison(rds, "addr") ++
        set_enable(rds, "en") ++
        set_poison(wrs, "addr") ++
        set_enable(wrs, "en") ++
        set_write(wrs, "data", "mask") ++
        set_poison(rws, "addr") ++
        set_wmode(rws, "wmode") ++
        set_enable(rws, "en") ++
        set_write(rws, "wdata", "wmask")
      val mem = DefMemory(s.info, s.name, s.tpe, s.size, 1, if (s.seq) 1 else 0,
                  rds map (_.name), wrs map (_.name), rws map (_.name))
      Block(mem +: stmts)
    case (s: CDefMPort) => {
      types(s.name) = types(s.mem)
      val addrs = ArrayBuffer[String]()
      val clks = ArrayBuffer[String]()
      val ens = ArrayBuffer[String]()
      s.direction match {
        case MReadWrite =>
          refs(s.name) = DataRef(SubField(Reference(s.mem, ut), s.name, ut), "rdata", "wdata", "wmask", true)
          addrs += "addr"
          clks += "clk"
          ens += "en"
        case MWrite =>
          refs(s.name) = DataRef(SubField(Reference(s.mem, ut), s.name, ut), "data", "data", "mask", false)
          addrs += "addr"
          clks += "clk"
          ens += "en"
        case MRead =>
          refs(s.name) = DataRef(SubField(Reference(s.mem, ut), s.name, ut), "data", "data", "blah", false)
          addrs += "addr"
          clks += "clk"
          s.exps.head match {
            case e: Reference if smems(s.mem) =>
              raddrs(e.name) = SubField(SubField(Reference(s.mem, ut), s.name, ut), "en", ut)
            case _ => ens += "en"
          }
      }
      Block(
        (addrs map (x => Connect(s.info, SubField(SubField(Reference(s.mem, ut), s.name, ut), x, ut), s.exps(0)))) ++
        (clks map (x => Connect(s.info, SubField(SubField(Reference(s.mem, ut), s.name, ut), x, ut), s.exps(1)))) ++
        (ens map (x => Connect(s.info,SubField(SubField(Reference(s.mem,ut), s.name, ut), x, ut), one))))
    }
    case (s) => s map collect_refs(mports, smems, types, refs, raddrs)
  }

  def get_mask(refs: DataRefMap)(e: Expression): Expression =
    e map get_mask(refs) match {
      case e: Reference => refs get e.name match {
        case None => e
        case Some(p) => SubField(p.exp, p.mask, createMask(e.tpe))
      }
      case e => e
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
      case e => e map remove_chirrtl_e(g)
   }
   (s) match {
      case DefNode(info, name, value) =>
        val valuex = remove_chirrtl_e(MALE)(value)
        val sx = DefNode(info, name, valuex)
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
      case s => s map remove_chirrtl_s(refs, raddrs) map remove_chirrtl_e(MALE)
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
    c copy (modules = (c.modules map remove_chirrtl_m))
}

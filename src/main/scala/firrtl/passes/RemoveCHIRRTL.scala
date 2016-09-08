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

import com.typesafe.scalalogging.LazyLogging
import java.nio.file.{Paths, Files}

// Datastructures
import scala.collection.mutable.LinkedHashMap
import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import scala.collection.mutable.ArrayBuffer

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import firrtl.PrimOps._
import firrtl.WrappedExpression._

case class MPort( val name : String, val clk : Expression)
case class MPorts( val readers : ArrayBuffer[MPort], val writers : ArrayBuffer[MPort], val readwriters : ArrayBuffer[MPort])
case class DataRef( val exp : Expression, val male : String, val female : String, val mask : String, val rdwrite : Boolean)

object RemoveCHIRRTL extends Pass {
  def name = "Remove CHIRRTL"
  var mname = ""
  def create_exps (e:Expression) : Seq[Expression] = e match {
    case (e:Mux) =>
      val e1s = create_exps(e.tval)
      val e2s = create_exps(e.fval)
      (e1s,e2s).zipped map ((e1,e2) => Mux(e.cond,e1,e2,mux_type(e1,e2)))
    case (e:ValidIf) =>
      create_exps(e.value) map (e1 => ValidIf(e.cond,e1,e1.tpe))
    case (e) => (e.tpe) match {
      case (_:GroundType) => Seq(e)
      case (t:BundleType) => (t.fields foldLeft Seq[Expression]())((exps, f) =>
         exps ++ create_exps(SubField(e,f.name,f.tpe)))
      case (t:VectorType) => ((0 until t.size) foldLeft Seq[Expression]())((exps, i) =>
         exps ++ create_exps(SubIndex(e,i,t.tpe)))
      case UnknownType => Seq(e)
    }
  }
  def run (c:Circuit) : Circuit = {
    def remove_chirrtl_m (m:Module) : Module = {
      val hash = LinkedHashMap[String,MPorts]()
      val repl = LinkedHashMap[String,DataRef]()
      val raddrs = HashMap[String, Expression]()
      val ut = UnknownType
      val mport_types = LinkedHashMap[String,Type]()
      val smems = HashSet[String]()
      def EMPs () : MPorts = MPorts(ArrayBuffer[MPort](),ArrayBuffer[MPort](),ArrayBuffer[MPort]())
      def collect_smems_and_mports (s:Statement) : Statement = {
        (s) match { 
          case (s:CDefMemory) if s.seq =>
             smems += s.name
             s
          case (s:CDefMPort) => {
            val mports = hash.getOrElse(s.mem,EMPs())
            s.direction match {
              case MRead => mports.readers += MPort(s.name,s.exps(1))
              case MWrite => mports.writers += MPort(s.name,s.exps(1))
              case MReadWrite => mports.readwriters += MPort(s.name,s.exps(1))
            }
            hash(s.mem) = mports
            s
          }
          case (s) => s map (collect_smems_and_mports)
        }
      }
      def collect_refs (s:Statement) : Statement = {
        (s) match { 
          case (s:CDefMemory) => {
            mport_types(s.name) = s.tpe
            val stmts = ArrayBuffer[Statement]()
            val taddr = UIntType(IntWidth(scala.math.max(1,ceil_log2(s.size))))
            val tdata = s.tpe
            def set_poison (vec:Seq[MPort],addr:String) : Unit = {
              for (r <- vec ) {
                stmts += IsInvalid(s.info,SubField(SubField(Reference(s.name,ut),r.name,ut),addr,taddr))
                stmts += IsInvalid(s.info,SubField(SubField(Reference(s.name,ut),r.name,ut),"clk",taddr))
              }
            }
            def set_enable (vec:Seq[MPort],en:String) : Unit = {
              for (r <- vec ) {
                stmts += Connect(s.info,SubField(SubField(Reference(s.name,ut),r.name,ut),en,taddr),zero)
              }}
            def set_wmode (vec:Seq[MPort],wmode:String) : Unit = {
              for (r <- vec) {
                stmts += Connect(s.info,SubField(SubField(Reference(s.name,ut),r.name,ut),wmode,taddr),zero)
              }}
            def set_write (vec:Seq[MPort],data:String,mask:String) : Unit = {
              val tmask = create_mask(s.tpe)
              for (r <- vec ) {
                stmts += IsInvalid(s.info,SubField(SubField(Reference(s.name,ut),r.name,ut),data,tdata))
                for (x <- create_exps(SubField(SubField(Reference(s.name,ut),r.name,ut),mask,tmask)) ) {
                  stmts += Connect(s.info,x,zero)
                }}}
            val rds = (hash.getOrElse(s.name,EMPs())).readers
            set_poison(rds,"addr")
            set_enable(rds,"en")
            val wrs = (hash.getOrElse(s.name,EMPs())).writers
            set_poison(wrs,"addr")
            set_enable(wrs,"en")
            set_write(wrs,"data","mask")
            val rws = (hash.getOrElse(s.name,EMPs())).readwriters
            set_poison(rws,"addr")
            set_wmode(rws,"wmode")
            set_enable(rws,"en")
            set_write(rws,"wdata","wmask")
            val read_l = if (s.seq) 1 else 0
            val mem = DefMemory(s.info,s.name,s.tpe,s.size,1,read_l,rds.map(_.name),wrs.map(_.name),rws.map(_.name))
            Block(Seq(mem,Block(stmts)))
          }
          case (s:CDefMPort) => {
            mport_types(s.name) = mport_types(s.mem)
            val addrs = ArrayBuffer[String]()
            val clks = ArrayBuffer[String]()
            val ens = ArrayBuffer[String]()
            val masks = ArrayBuffer[String]()
            s.direction match {
              case MReadWrite => {
                repl(s.name) = DataRef(SubField(Reference(s.mem,ut),s.name,ut),"rdata","wdata","wmask",true)
                addrs += "addr"
                clks += "clk"
                ens += "en"
                masks += "wmask"
              }
              case MWrite => {
                repl(s.name) = DataRef(SubField(Reference(s.mem,ut),s.name,ut),"data","data","mask",false)
                addrs += "addr"
                clks += "clk"
                ens += "en"
                masks += "mask"
              }
              case MRead => {
                repl(s.name) = DataRef(SubField(Reference(s.mem,ut),s.name,ut),"data","data","blah",false)
                addrs += "addr"
                clks += "clk"
                s.exps(0) match {
                   case e: Reference if smems(s.mem) =>
                      raddrs(e.name) = SubField(SubField(Reference(s.mem,ut),s.name,ut),"en",ut)
                   case _ => ens += "en"
                }
              }
            }
            val stmts = ArrayBuffer[Statement]()
            for (x <- addrs ) {
               stmts += Connect(s.info,SubField(SubField(Reference(s.mem,ut),s.name,ut),x,ut),s.exps(0))
            }
            for (x <- clks ) {
              stmts += Connect(s.info,SubField(SubField(Reference(s.mem,ut),s.name,ut),x,ut),s.exps(1))
            }
            for (x <- ens ) {
              stmts += Connect(s.info,SubField(SubField(Reference(s.mem,ut),s.name,ut),x,ut),one)
            }
            Block(stmts)
          }
          case (s) => s map (collect_refs)
        }
      }
      def remove_chirrtl_s (s:Statement) : Statement = {
        var has_write_mport = false
        var has_read_mport: Option[Expression] = None
        var has_readwrite_mport: Option[Expression] = None
        def remove_chirrtl_e (g:Gender)(e:Expression) : Expression = {
          (e) match {
            case (e:Reference) if repl contains e.name =>
              val vt = repl(e.name)
              g match {
                case MALE => SubField(vt.exp,vt.male,e.tpe)
                case FEMALE => {
                  has_write_mport = true
                  if (vt.rdwrite) 
                    has_readwrite_mport = Some(SubField(vt.exp,"wmode",UIntType(IntWidth(1))))
                  SubField(vt.exp,vt.female,e.tpe)
                }
              }
            case (e:Reference) if g == FEMALE && (raddrs contains e.name) =>
              has_read_mport = Some(raddrs(e.name))
              e
            case (e:Reference) => e
            case (e:SubAccess) => SubAccess(remove_chirrtl_e(g)(e.expr),remove_chirrtl_e(MALE)(e.index),e.tpe)
            case (e) => e map (remove_chirrtl_e(g))
          }
        }
        def get_mask (e:Expression) : Expression = {
          (e map (get_mask)) match { 
            case (e:Reference) => {
              if (repl.contains(e.name)) {
                val vt = repl(e.name)
                val t = create_mask(e.tpe)
                SubField(vt.exp,vt.mask,t)
              } else e
            }
            case (e) => e
          }
        }
        (s) match {
          case (s:DefNode) => {
            val stmts = ArrayBuffer[Statement]()
            val valuex = remove_chirrtl_e(MALE)(s.value)
            stmts += DefNode(s.info,s.name,valuex)
            has_read_mport match {
              case None =>
              case Some(en) => stmts += Connect(s.info,en,one)
            }
            if (stmts.size > 1) Block(stmts)
            else stmts(0)
          }
          case (s:Connect) => {
            val stmts = ArrayBuffer[Statement]()
            val rocx = remove_chirrtl_e(MALE)(s.expr)
            val locx = remove_chirrtl_e(FEMALE)(s.loc)
            stmts += Connect(s.info,locx,rocx)
            has_read_mport match {
              case None =>
              case Some(en) => stmts += Connect(s.info,en,one)
            }
            if (has_write_mport) {
              val e = get_mask(s.loc)
              for (x <- create_exps(e) ) {
                stmts += Connect(s.info,x,one)
              }
              has_readwrite_mport match {
                case None =>
                case Some(wmode) => stmts += Connect(s.info,wmode,one)
              }
            }
            if (stmts.size > 1) Block(stmts)
            else stmts(0)
          }
          case (s:PartialConnect) => {
            val stmts = ArrayBuffer[Statement]()
            val locx = remove_chirrtl_e(FEMALE)(s.loc)
            val rocx = remove_chirrtl_e(MALE)(s.expr)
            stmts += PartialConnect(s.info,locx,rocx)
            has_read_mport match {
              case None =>
              case Some(en) => stmts += Connect(s.info,en,one)
            }
            if (has_write_mport) {
              val ls = get_valid_points(s.loc.tpe,s.expr.tpe,Default,Default)
              val locs = create_exps(get_mask(s.loc))
              for (x <- ls ) {
                val locx = locs(x._1)
                stmts += Connect(s.info,locx,one)
              }
              has_readwrite_mport match {
                case None =>
                case Some(wmode) => stmts += Connect(s.info,wmode,one)
              }
            }
            if (stmts.size > 1) Block(stmts)
            else stmts(0)
          }
          case (s) => s map (remove_chirrtl_s) map (remove_chirrtl_e(MALE))
        }
      }
      collect_smems_and_mports(m.body)
      val sx = collect_refs(m.body)
      Module(m.info,m.name, m.ports, remove_chirrtl_s(sx))
    }
    val modulesx = c.modules.map{ m => {
      (m) match { 
        case (m:Module) => remove_chirrtl_m(m)
        case (m:ExtModule) => m
      }}}
    Circuit(c.info,modulesx, c.main)
  }
}

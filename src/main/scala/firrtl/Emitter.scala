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

package firrtl

import com.typesafe.scalalogging.LazyLogging
import java.nio.file.{Paths, Files}
import java.io.Writer
import java.io.Reader

import scala.sys.process._
import scala.io.Source

import Utils._
import firrtl.Serialize._
import firrtl.Mappers._
import firrtl.passes._
import firrtl.PrimOps._
import firrtl.ir._
import WrappedExpression._
// Datastructures
import scala.collection.mutable.LinkedHashMap
import scala.collection.mutable.ArrayBuffer

class EmitterException(message: String) extends PassException(message)

trait Emitter extends LazyLogging {
  def run(c: Circuit, w: Writer)
}

object FIRRTLEmitter extends Emitter {
  def run(c: Circuit, w: Writer) = w.write(c.serialize)
}

case class VIndent()
case object VRandom extends Expression {
  def tpe = UIntType(UnknownWidth)
}
class VerilogEmitter extends Emitter {
   val tab = "  "
   var w:Option[Writer] = None
   var mname = ""
   def wref (n:String,t:Type) = WRef(n,t,ExpKind(),UNKNOWNGENDER)
   def remove_root (ex:Expression) : Expression = {
      (ex.as[WSubField].get.exp) match {
         case (e:WSubField) => remove_root(e)
         case (e:WRef) => WRef(ex.as[WSubField].get.name,tpe(ex),InstanceKind(),UNKNOWNGENDER)
      }
   }
   def not_empty (s:ArrayBuffer[_]) : Boolean = if (s.size == 0) false else true
   def emit (x:Any) = emit2(x,0)
   def emit2 (x:Any, top:Int) : Unit = {
      def cast (e:Expression) : Any = {
         e.tpe match {
            case (t:UIntType) => e
            case (t:SIntType) => Seq("$signed(",e,")")
            case ClockType => e
         }
      }
      (x) match {
         case (e:Expression) => {
            (e) match {
               case (e:DoPrim) => emit2(op_stream(e), top + 1)
               case (e:Mux) => emit2(Seq(e.cond," ? ",cast(e.tval)," : ",cast(e.fval)),top + 1)
               case (e:ValidIf) => emit2(Seq(cast(e.value)),top + 1)
               case (e:WRef) => w.get.write(e.serialize)
               case (e:WSubField) => w.get.write(LowerTypes.loweredName(e))
               case (e:WSubAccess) => w.get.write(LowerTypes.loweredName(e.exp) + "[" + LowerTypes.loweredName(e.index) + "]")
               case (e:WSubIndex) => w.get.write(e.serialize)
               case (e:Literal) => v_print(e)
               case VRandom => w.get.write("$random")
            }
         }
         case (t:Type) => {
            (t) match {
               case (_:UIntType|_:SIntType) => 
                  val wx = long_BANG(t) - 1
                  if (wx > 0) w.get.write("[" + wx + ":0]") else w.get.write("")
               case ClockType => w.get.write("")
               case (t:VectorType) => 
                  emit2(t.tpe, top + 1)
                  w.get.write("[" + (t.size - 1) + ":0]")
               case (t) => error("Shouldn't be here"); w.get.write(t.serialize)
            }
         }
         case (p:Direction) => {
            p match {
               case Input => w.get.write("input")
               case Output => w.get.write("output")
            }
         }
         case (s:String) => w.get.write(s)
         case (i:Int) => w.get.write(i.toString)
         case (i:Long) => w.get.write(i.toString)
         case (t:VIndent) => w.get.write("   ")
         case (s:Seq[Any]) => {
            s.foreach((x:Any) => emit2(x.as[Any].get, top + 1))
            if (top == 0) w.get.write("\n")
         }
      }
   }

   //;------------- PASS -----------------
   def v_print (e:Expression) = {
      e match {
         case (e:UIntLiteral) => {
            val str = e.value.toString(16)
            w.get.write(long_BANG(tpe(e)).toString + "'h" + str)
         }
         case (e:SIntLiteral) => {
            val str = e.value.toString(16)
            w.get.write(long_BANG(tpe(e)).toString + "'sh" + str)
         }
      }
   }
   def op_stream (doprim:DoPrim) : Seq[Any] = {
      def cast_if (e:Expression) : Any = {
         val signed = doprim.args.find(x => tpe(x).typeof[SIntType])
         if (signed == None) e
         else tpe(e) match {
            case (t:SIntType) => Seq("$signed(",e,")")
            case (t:UIntType) => Seq("$signed({1'b0,",e,"})")
         }
      }
      def cast (e:Expression) : Any = {
         (doprim.tpe) match {
            case (t:UIntType) => e
            case (t:SIntType) => Seq("$signed(",e,")")
         }
      }
      def cast_as (e:Expression) : Any = {
         (tpe(e)) match {
            case (t:UIntType) => e
            case (t:SIntType) => Seq("$signed(",e,")")
         }
      }
      def a0 () : Expression = doprim.args(0)
      def a1 () : Expression = doprim.args(1)
      def a2 () : Expression = doprim.args(2)
      def c0 () : Int = doprim.consts(0).toInt
      def c1 () : Int = doprim.consts(1).toInt

      def checkArgumentLegality(e: Expression) = e match {
        case _: UIntLiteral =>
        case _: SIntLiteral =>
        case _: WRef =>
        case _: WSubField =>
        case _ => throw new EmitterException(s"Can't emit ${e.getClass.getName} as PrimOp argument")
      }

      doprim.args foreach checkArgumentLegality
   
      doprim.op match {
         case Add => Seq(cast_if(a0())," + ", cast_if(a1()))
         case Addw => Seq(cast_if(a0())," + ", cast_if(a1()))
         case Sub => Seq(cast_if(a0())," - ", cast_if(a1()))
         case Subw => Seq(cast_if(a0())," - ", cast_if(a1()))
         case Mul => Seq(cast_if(a0())," * ", cast_if(a1()) )
         case Div => Seq(cast_if(a0())," / ", cast_if(a1()) )
         case Rem => Seq(cast_if(a0())," % ", cast_if(a1()) )
         case Lt => Seq(cast_if(a0())," < ", cast_if(a1()))
         case Leq => Seq(cast_if(a0())," <= ", cast_if(a1()))
         case Gt => Seq(cast_if(a0())," > ", cast_if(a1()))
         case Geq => Seq(cast_if(a0())," >= ", cast_if(a1()))
         case Eq => Seq(cast_if(a0())," == ", cast_if(a1()))
         case Neq => Seq(cast_if(a0())," != ", cast_if(a1()))
         case Pad => {
            val w = long_BANG(tpe(a0()))
            val diff = (c0() - w)
            if (w == 0) Seq(a0())
            else doprim.tpe match {
               // Either sign extend or zero extend.
               case (t:SIntType) => {
                  // If width == 1, don't extract bit
                  if (w == 1) Seq("{", c0(),"{", a0(), "}}")
                  else Seq("{{", diff, "{", a0(), "[", w - 1,"]}},", a0(), "}")
               }
               case (t) => Seq("{{", diff, "'d0}, ", a0(), "}")
            }
         }
         case AsUInt => Seq("$unsigned(",a0(),")")
         case AsSInt => Seq("$signed(",a0(),")")
         case AsClock => Seq("$unsigned(",a0(),")")
         case Dshlw => Seq(cast(a0())," << ", a1())
         case Dshl => Seq(cast(a0())," << ", a1())
         case Dshr => {
            (doprim.tpe) match {
               case (t:SIntType) => Seq(cast(a0())," >>> ",a1())
               case (t) => Seq(cast(a0())," >> ",a1())
            }
         }
         case Shlw => Seq(cast(a0())," << ", c0())
         case Shl => Seq(cast(a0())," << ",c0())
         case Shr => {
           if (c0 >= long_BANG(tpe(a0)))
             error("Verilog emitter does not support SHIFT_RIGHT >= arg width")
           Seq(a0(),"[", long_BANG(tpe(a0())) - 1,":",c0(),"]")
         }
         case Neg => Seq("-{",cast(a0()),"}")
         case Cvt => {
            tpe(a0()) match {
               case (t:UIntType) => Seq("{1'b0,",cast(a0()),"}")
               case (t:SIntType) => Seq(cast(a0()))
            }
         }
         case Not => Seq("~ ",a0())
         case And => Seq(cast_as(a0())," & ", cast_as(a1()))
         case Or => Seq(cast_as(a0())," | ", cast_as(a1()))
         case Xor => Seq(cast_as(a0())," ^ ", cast_as(a1()))
         case Andr => {
            val v = ArrayBuffer[Seq[Any]]()
            for (b <- 0 until long_BANG(doprim.tpe).toInt) {
               v += Seq(cast(a0()),"[",b,"]")
            }
            v.reduce(_ + " & " + _)
         }
         case Orr => {
            val v = ArrayBuffer[Seq[Any]]()
            for (b <- 0 until long_BANG(doprim.tpe).toInt) {
               v += Seq(cast(a0()),"[",b,"]")
            }
            v.reduce(_ + " | " + _)
         }
         case Xorr => {
            val v = ArrayBuffer[Seq[Any]]()
            for (b <- 0 until long_BANG(doprim.tpe).toInt) {
               v += Seq(cast(a0()),"[",b,"]")
            }
            v.reduce(_ + " ^ " + _)
         }
         case Cat => Seq("{",cast(a0()),",",cast(a1()),"}")
         case Bits => {
            // If selecting zeroth bit and single-bit wire, just emit the wire
            if (c0() == 0 && c1() == 0 && long_BANG(tpe(a0())) == 1) Seq(a0())
            else if (c0() == c1()) Seq(a0(),"[",c0(),"]")
            else Seq(a0(),"[",c0(),":",c1(),"]")
         }
         case Head => {
            val w = long_BANG(tpe(a0()))
            val high = w - 1
            val low = w - c0()
            Seq(a0(),"[",high,":",low,"]")
         }
         case Tail => {
            val w = long_BANG(tpe(a0()))
            val low = w - c0() - 1
            Seq(a0(),"[",low,":",0,"]")
         }
      }
   }
   
   def emit_verilog (m:Module) : DefModule = {
      mname = m.name
      val netlist = LinkedHashMap[WrappedExpression,Expression]()
      val simlist = ArrayBuffer[Statement]()
      val namespace = Namespace(m)
      def build_netlist (s:Statement) : Statement = {
         s match {
            case (s:Connect) => netlist(s.loc) = s.expr
            case (s:IsInvalid) => {
               val n = namespace.newTemp
               val e = wref(n,tpe(s.expr))
               netlist(s.expr) = e
            }
            case (s:Conditionally) => simlist += s
            case (s:DefNode) => {
               val e = WRef(s.name,get_type(s),NodeKind(),MALE)
               netlist(e) = s.value
            }
            case (s) => s map (build_netlist)
         }
         s
      }
   
      val portdefs = ArrayBuffer[Seq[Any]]()
      val declares = ArrayBuffer[Seq[Any]]()
      val instdeclares = ArrayBuffer[Seq[Any]]()
      val assigns = ArrayBuffer[Seq[Any]]()
      val at_clock = LinkedHashMap[Expression,ArrayBuffer[Seq[Any]]]()
      val initials = ArrayBuffer[Seq[Any]]()
      val simulates = ArrayBuffer[Seq[Any]]()
      def declare (b:String,n:String,t:Type) = {
         t match {
            case (t:VectorType) => declares += Seq(b," ",t.tpe," ",n," [0:",t.size - 1,"];")
            case (t) => declares += Seq(b," ",t," ",n,";")
         }
      }
      def assign (e:Expression,value:Expression) =
         assigns += Seq("assign ",e," = ",value,";")
      // Like assign but with different versions for synthesis and simulation
      def synSimAssign(e: Expression, syn: Expression, sim: Expression) = {
         assigns += Seq("`ifdef SYNTHESIS")
         assigns += Seq("assign ", e, " = ", syn, ";")
         assigns += Seq("`else")
         assigns += Seq("assign ", e, " = ", sim, ";")
         assigns += Seq("`endif")
      }
      def update_and_reset(r: Expression, clk: Expression, reset: Expression, init: Expression) = {
        // We want to flatten Mux trees for reg updates into if-trees for
        // improved QoR for conditional updates.  However, unbounded recursion
        // would take exponential time, so don't redundantly flatten the same
        // Mux more than a bounded number of times, preserving linear runtime.
        // The threshold is empirical but ample.
        val flattenThreshold = 4
        val numTimesFlattened = collection.mutable.HashMap[Mux, Int]()
        def canFlatten(m: Mux) = {
          val n = numTimesFlattened.getOrElse(m, 0)
          numTimesFlattened(m) = n + 1
          n < flattenThreshold
        }

        def addUpdate(e: Expression, tabs: String): Seq[Seq[Any]] = {
          netlist.getOrElse(e, e) match {
            case m: Mux if canFlatten(m) => {
              val ifStatement = Seq(tabs, "if(", m.cond, ") begin")
              val trueCase = addUpdate(m.tval, tabs + tab)
              val elseStatement = Seq(tabs, "end else begin")
              val falseCase = addUpdate(m.fval, tabs + tab)
              val endStatement = Seq(tabs, "end")

              if (falseCase.isEmpty)
                ifStatement +: trueCase :+ endStatement
              else
                ifStatement +: trueCase ++: elseStatement +: falseCase :+ endStatement
            }
            case _ if (weq(e, r)) => Seq()
            case _ => Seq(Seq(tabs, r, " <= ", e, ";"))
          }
        }

        at_clock.getOrElseUpdate(clk, ArrayBuffer[Seq[Any]]()) ++= {
          val tv = init
          val fv = netlist(r)
          addUpdate(Mux(reset, tv, fv, mux_type_and_widths(tv, fv)), "")
        }
      }
      def update (e:Expression,value:Expression,clk:Expression,en:Expression) = {
         if (!at_clock.contains(clk)) at_clock(clk) = ArrayBuffer[Seq[Any]]()
         if (weq(en,one)) at_clock(clk) += Seq(e," <= ",value,";")
         else {
            at_clock(clk) += Seq("if(",en,") begin")
            at_clock(clk) += Seq(tab,e," <= ",value,";")
            at_clock(clk) += Seq("end")
         }
      }
      // Declares an intermediate wire to hold a large enough random number.
      // Then, return the correct number of bits selected from the random value
      def rand_string (t:Type) : Seq[Any] = {
         val nx = namespace.newTemp
         val wx = ((long_BANG(t) + 31) / 32).toInt
         val tx = SIntType(IntWidth(wx * 32))
         val rand = Seq("{",wx.toString,"{",VRandom,"}}")
         declare("reg",nx,tx)
         initials += Seq(wref(nx,tx)," = ",rand,";")
         Seq(nx,"[",long_BANG(t) - 1,":0]")
      }
      def initialize (e:Expression) = initials += Seq(e," = ",rand_string(tpe(e)),";")
      def initialize_mem(s: DefMemory) = {
        val index = WRef("initvar", s.dataType, ExpKind(), UNKNOWNGENDER)
        val rstring = rand_string(s.dataType)
        initials += Seq("for (initvar = 0; initvar < ", s.depth, "; initvar = initvar+1)")
        initials += Seq(tab, WSubAccess(wref(s.name, s.dataType), index, s.dataType, FEMALE), " = ", rstring,";")
      }
      def instantiate (n:String,m:String,es:Seq[Expression]) = {
         instdeclares += Seq(m," ",n," (")
         (es,0 until es.size).zipped.foreach{ (e,i) => {
            val s = Seq(tab,".",remove_root(e),"(",LowerTypes.loweredName(e),")")
            if (i != es.size - 1) instdeclares += Seq(s,",")
            else instdeclares += s
         }}
         instdeclares += Seq(");")
         for (e <- es) {
            declare("wire",LowerTypes.loweredName(e),tpe(e))
            val ex = WRef(LowerTypes.loweredName(e),tpe(e),kind(e),gender(e))
            if (gender(e) == FEMALE) {
               assign(ex,netlist(e))
            }
         }
      }
      def simulate(clk: Expression, en: Expression, s: Seq[Any], cond: Option[String]) = {
        if (!at_clock.contains(clk)) at_clock(clk) = ArrayBuffer[Seq[Any]]()
        at_clock(clk) += Seq("`ifndef SYNTHESIS")
        if (cond.nonEmpty) {
          at_clock(clk) += Seq(s"`ifdef ${cond.get}")
          at_clock(clk) += Seq(tab, s"if (`${cond.get}) begin")
          at_clock(clk) += Seq("`endif")
        }
        at_clock(clk) += Seq(tab,tab,"if (",en,") begin")
        at_clock(clk) += Seq(tab,tab,tab,s)
        at_clock(clk) += Seq(tab,tab,"end")
        if (cond.nonEmpty) {
          at_clock(clk) += Seq(s"`ifdef ${cond.get}")
          at_clock(clk) += Seq(tab,"end")
          at_clock(clk) += Seq("`endif")
        }
        at_clock(clk) += Seq("`endif")
      }
      def stop(ret: Int): Seq[Any] = {
        Seq(if (ret == 0) "$finish;" else "$fatal;")
      }
      def printf (str:StringLit,args:Seq[Expression]) : Seq[Any] = {
         val q = '"'.toString
	       val strx = Seq(q + VerilogStringLitHandler.escape(str) + q) ++
                    args.flatMap(x => Seq(",",x))
         Seq("$fwrite(32'h80000002,",strx,");")
      }
      def delay (e:Expression, n:Int, clk:Expression) : Expression = {
         var ex = e
         for (i <- 0 until n) {
            val name = namespace.newTemp
            declare("reg",name,tpe(e))
            val exx = WRef(name,tpe(e),ExpKind(),UNKNOWNGENDER)
            initialize(exx)
            update(exx,ex,clk,one)
            ex = exx
         }
         ex
      }
      def build_ports () = {
         (m.ports,0 until m.ports.size).zipped.foreach{(p,i) => {
            p.direction match {
               case Input => portdefs += Seq(p.direction,"  ",p.tpe," ",p.name)
               case Output => {
                  portdefs += Seq(p.direction," ",p.tpe," ",p.name)
                  val ex = WRef(p.name,p.tpe,PortKind(),FEMALE)
                  assign(ex,netlist(ex))
               }
            }
         }}
      }
      def build_streams (s:Statement) : Statement = {
         s match {
            case EmptyStmt => s
            case (s:Connect) => s
            case (s:DefWire) => 
               declare("wire",s.name,s.tpe)
               val e = wref(s.name,s.tpe)
               assign(e,netlist(e))
            case (s:DefRegister) => {
               declare("reg",s.name,s.tpe)
               val e = wref(s.name,s.tpe)
               update_and_reset(e,s.clock,s.reset,s.init)
               initialize(e)
            }
            case (s:IsInvalid) => {
               val wref = netlist(s.expr).as[WRef].get
               declare("reg",wref.name,tpe(s.expr))
               initialize(wref)
            }
            case (s:DefNode) => {
               declare("wire",s.name,tpe(s.value))
               assign(WRef(s.name,tpe(s.value),NodeKind(),MALE),s.value)
            }
            case (s:Stop) => {
              val errorString = StringLit(s"${s.ret}\n".getBytes)
              build_streams(Print(NoInfo, errorString, Seq(), s.clk, s.en))
              simulate(s.clk, s.en, stop(s.ret), Some("STOP_COND"))
            }
            case (s:Print) => simulate(s.clk, s.en, printf(s.string, s.args), Some("PRINTF_COND"))
            case (s:WDefInstance) => {
               val es = create_exps(WRef(s.name,s.tpe,InstanceKind(),MALE))
               instantiate(s.name,s.module,es)
            }
            case (s:DefMemory) => {
               val mem = WRef(s.name,get_type(s),MemKind(s.readers ++ s.writers ++ s.readwriters),UNKNOWNGENDER)
               def mem_exp (p:String,f:String) = {
                  val t1 = field_type(mem.tpe,p)
                  val t2 = field_type(t1,f)
                  val x = WSubField(mem,p,t1,UNKNOWNGENDER)
                  WSubField(x,f,t2,UNKNOWNGENDER)
               }
      
               declare("reg",s.name,VectorType(s.dataType,s.depth))
               initialize_mem(s)
               for (r <- s.readers ) {
                  val data = mem_exp(r,"data")
                  val addr = mem_exp(r,"addr")
                  val en = mem_exp(r,"en")
                  //Ports should share an always@posedge, so can't have intermediary wire
                  val clk = netlist(mem_exp(r,"clk")) 
                  
                  declare("wire",LowerTypes.loweredName(data),tpe(data))
                  declare("wire",LowerTypes.loweredName(addr),tpe(addr))
                  declare("wire",LowerTypes.loweredName(en),tpe(en))
   
                  //; Read port
                  assign(addr,netlist(addr)) //;Connects value to m.r.addr
                  assign(en,netlist(en))     //;Connects value to m.r.en
                  val addrx = delay(addr,s.readLatency,clk)
                  val enx = delay(en,s.readLatency,clk)
                  val mem_port = WSubAccess(mem,addrx,s.dataType,UNKNOWNGENDER)
                  val depthValue = UIntLiteral(s.depth, IntWidth(BigInt(s.depth).bitLength))
                  val garbageGuard = DoPrim(Geq, Seq(addrx, depthValue), Seq(), UnknownType)
                  val garbageMux = Mux(garbageGuard, VRandom, mem_port, UnknownType)
                  synSimAssign(data, mem_port, garbageMux)
               }
   
               for (w <- s.writers ) {
                  val data = mem_exp(w,"data")
                  val addr = mem_exp(w,"addr")
                  val mask = mem_exp(w,"mask")
                  val en = mem_exp(w,"en")
                  //Ports should share an always@posedge, so can't have intermediary wire
                  val clk = netlist(mem_exp(w,"clk"))
                  
                  declare("wire",LowerTypes.loweredName(data),tpe(data))
                  declare("wire",LowerTypes.loweredName(addr),tpe(addr))
                  declare("wire",LowerTypes.loweredName(mask),tpe(mask))
                  declare("wire",LowerTypes.loweredName(en),tpe(en))
   
                  //; Write port
                  assign(data,netlist(data))
                  assign(addr,netlist(addr))
                  assign(mask,netlist(mask))
                  assign(en,netlist(en))
   
                  val datax = delay(data,s.writeLatency - 1,clk)
                  val addrx = delay(addr,s.writeLatency - 1,clk)
                  val maskx = delay(mask,s.writeLatency - 1,clk)
                  val enx = delay(en,s.writeLatency - 1,clk)
                  val mem_port = WSubAccess(mem,addrx,s.dataType,UNKNOWNGENDER)
                  update(mem_port,datax,clk,AND(enx,maskx))
               }
   
               for (rw <- s.readwriters) {
                  val wmode = mem_exp(rw,"wmode")
                  val rdata = mem_exp(rw,"rdata")
                  val data = mem_exp(rw,"data")
                  val mask = mem_exp(rw,"mask")
                  val addr = mem_exp(rw,"addr")
                  val en = mem_exp(rw,"en")
                  //Ports should share an always@posedge, so can't have intermediary wire
                  val clk = netlist(mem_exp(rw,"clk"))
                  
                  declare("wire",LowerTypes.loweredName(wmode),tpe(wmode))
                  declare("wire",LowerTypes.loweredName(rdata),tpe(rdata))
                  declare("wire",LowerTypes.loweredName(data),tpe(data))
                  declare("wire",LowerTypes.loweredName(mask),tpe(mask))
                  declare("wire",LowerTypes.loweredName(addr),tpe(addr))
                  declare("wire",LowerTypes.loweredName(en),tpe(en))
   
                  //; Assigned to lowered wires of each
                  assign(addr,netlist(addr))
                  assign(data,netlist(data))
                  assign(addr,netlist(addr))
                  assign(mask,netlist(mask))
                  assign(en,netlist(en))
                  assign(wmode,netlist(wmode))
   
                  //; Delay new signals by latency
                  val raddrx = delay(addr,s.readLatency,clk)
                  val waddrx = delay(addr,s.writeLatency - 1,clk)
                  val enx = delay(en,s.writeLatency - 1,clk)
                  val rmodx = delay(wmode,s.writeLatency - 1,clk)
                  val datax = delay(data,s.writeLatency - 1,clk)
                  val maskx = delay(mask,s.writeLatency - 1,clk)
   
                  //; Write 
   
                  val rmem_port = WSubAccess(mem,raddrx,s.dataType,UNKNOWNGENDER)
                  assign(rdata,rmem_port)
                  val wmem_port = WSubAccess(mem,waddrx,s.dataType,UNKNOWNGENDER)

                  val tempName = namespace.newTemp
                  val tempExp = AND(enx,maskx)
                  declare("wire", tempName, tpe(tempExp))
                  val tempWRef = wref(tempName, tpe(tempExp))
                  assign(tempWRef, tempExp)
                  update(wmem_port,datax,clk,AND(tempWRef,wmode))
               }
            }
            case (s:Begin) => s map (build_streams)
         }
         s
      }
   
      def emit_streams () = {
         emit(Seq("module ",m.name,"("))
         if (not_empty(portdefs)) {
            (portdefs,0 until portdefs.size).zipped.foreach{ (x,i) => {
               if (i != portdefs.size - 1) emit(Seq(tab,x,","))
               else emit(Seq(tab,x))
            }}
         }
         emit(Seq(");"))

         if (not_empty(declares)) {
            for (x <- declares) emit(Seq(tab,x))
         }
         if (not_empty(instdeclares)) {
            for (x <- instdeclares) emit(Seq(tab,x))
         }
         if (not_empty(assigns)) {
            for (x <- assigns) emit(Seq(tab,x))
         }
         if (not_empty(initials)) {
            emit(Seq("`ifndef SYNTHESIS"))
            emit(Seq("  integer initvar;"))
            emit(Seq("  initial begin"))
            // This enables test benches to set the random values at time 0.001,
            //  then start the simulation later
            // Verilator does not support delay statements, so they are omitted.
            emit(Seq("    `ifndef verilator"))
            emit(Seq("      #0.002;"))
            emit(Seq("    `endif"))
            for (x <- initials) {
               emit(Seq(tab,x))
            }
            emit(Seq("  end"))
            emit(Seq("`endif"))
         }
   
         for (clk_stream <- at_clock) {
            if (not_empty(clk_stream._2)) {
               emit(Seq(tab,"always @(posedge ",clk_stream._1,") begin"))
               for (x <- clk_stream._2) {
                  emit(Seq(tab,tab,x))
               }
               emit(Seq(tab,"end"))
            }
         }
   
         emit(Seq("endmodule"))
      }
   
      build_netlist(m.body)
      build_ports()
      build_streams(m.body)
      emit_streams()
      m
   }
   
   def run(c: Circuit, w: Writer) = {
      this.w = Some(w)
      for (m <- c.modules) {
         m match {
            case (m:Module) => emit_verilog(m)
            case (m:ExtModule) => false
         }
      }
   }
  //def run(c: Circuit, w: Writer) 
  //{
  //  logger.debug(s"Verilog Emitter is not yet implemented in Scala")
  //  val toStanza = Files.createTempFile(Paths.get(""), "verilog", ".fir")
  //  val fromStanza = Files.createTempFile(Paths.get(""), "verilog", ".fir")
  //  Files.write(toStanza, c.serialize.getBytes)

  //  val cmd = Seq("firrtl-stanza", "-i", toStanza.toString, "-o", fromStanza.toString, "-b", "verilog")
  //  logger.debug(cmd.mkString(" "))
  //  val ret = cmd.!
  //  // Copy from Stanza output to user requested outputFile (we can't get filename from Writer)
  //  Source.fromFile(fromStanza.toString) foreach { w.write(_) }

  //  Files.delete(toStanza)
  //  Files.delete(fromStanza)
  //}
}

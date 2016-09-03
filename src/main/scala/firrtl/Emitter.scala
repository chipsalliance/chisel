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
import java.io.{Reader, Writer}

import scala.sys.process._
import scala.io.Source

import Utils._
import firrtl.ir._
import firrtl.passes._
import firrtl.Mappers._
import firrtl.PrimOps._
import firrtl.WrappedExpression._
// Datastructures
import scala.collection.mutable.{ArrayBuffer, LinkedHashMap}

class EmitterException(message: String) extends PassException(message)

trait Emitter extends LazyLogging {
  def run(c: Circuit, w: Writer)
}

object FIRRTLEmitter extends Emitter {
  def run(c: Circuit, w: Writer) = w.write(c.serialize)
}

case class VIndent()
case class VRandom(width: BigInt) extends Expression {
  def tpe = UIntType(IntWidth(width))
  def nWords = (width + 31) / 32
  def realWidth = nWords * 32
  def serialize: String = "RANDOM"
  def mapExpr(f: Expression => Expression): Expression = this
  def mapType(f: Type => Type): Expression = this
  def mapWidth(f: Width => Width): Expression = this
}
class VerilogEmitter extends Emitter {
  val tab = "  "
  def AND(e1: WrappedExpression, e2: WrappedExpression): Expression = {
    if (e1 == e2) e1.e1
    else if ((e1 == we(zero)) | (e2 == we(zero))) zero
    else if (e1 == we(one)) e2.e1
    else if (e2 == we(one)) e1.e1
    else DoPrim(And, Seq(e1.e1, e2.e1), Nil, UIntType(IntWidth(1)))
  }
  def OR(e1: WrappedExpression, e2: WrappedExpression): Expression = {
    if (e1 == e2) e1.e1
    else if ((e1 == we(one)) | (e2 == we(one))) one
    else if (e1 == we(zero)) e2.e1
    else if (e2 == we(zero)) e1.e1
    else DoPrim(Or, Seq(e1.e1, e2.e1), Nil, UIntType(IntWidth(1)))
  }
  def NOT(e: WrappedExpression): Expression = {
    if (e == we(one)) zero
    else if (e == we(zero)) one
    else DoPrim(Eq, Seq(e.e1, zero), Nil, UIntType(IntWidth(1)))
  }

  def wref(n: String, t: Type) = WRef(n, t, ExpKind, UNKNOWNGENDER)
  def remove_root(ex: Expression): Expression = ex match {
    case ex: WSubField => ex.exp match {
      case (e: WSubField) => remove_root(e)
      case (_: WRef) => WRef(ex.name, ex.tpe, InstanceKind, UNKNOWNGENDER)
    }
    case _ => error("Shouldn't be here")
  }
  def emit(x: Any)(implicit w: Writer) { emit(x, 0) }
  def emit(x: Any, top: Int)(implicit w: Writer) {
    def cast(e: Expression): Any = e.tpe match {
      case (t: UIntType) => e
      case (t: SIntType) => Seq("$signed(",e,")")
      case ClockType => e
    }
    (x) match {
      case (e: DoPrim) => emit(op_stream(e), top + 1)
      case (e: Mux) => emit(Seq(e.cond," ? ",cast(e.tval)," : ",cast(e.fval)),top + 1)
      case (e: ValidIf) => emit(Seq(cast(e.value)),top + 1)
      case (e: WRef) => w write e.serialize
      case (e: WSubField) => w write LowerTypes.loweredName(e)
      case (e: WSubAccess) => w write (
        s"${LowerTypes.loweredName(e.exp)}[${LowerTypes.loweredName(e.index)}]")
      case (e: WSubIndex) => w write e.serialize
      case (e: Literal) => v_print(e)
      case (e: VRandom) => w write s"{${e.nWords}{$$random}}"
      case (t: UIntType) => 
        val wx = bitWidth(t) - 1
        if (wx > 0) w write s"[$wx:0]"
      case (t: SIntType) => 
        val wx = bitWidth(t) - 1
        if (wx > 0) w write s"[$wx:0]"
      case ClockType =>
      case (t: VectorType) => 
        emit(t.tpe, top + 1)
        w write s"[${t.size - 1}:0]"
      case Input => w write "input"
      case Output => w write "output"
      case (s: String) => w write s
      case (i: Int) => w write i.toString
      case (i: Long) => w write i.toString
      case (i: BigInt) => w write i.toString
      case (t: VIndent) => w write "   "
      case (s: Seq[Any]) =>
        s foreach (emit(_, top + 1))
        if (top == 0) w write "\n"
      case _ => error("Shouldn't be here")
    }
  }

   //;------------- PASS -----------------
   def v_print(e: Expression)(implicit w: Writer) = e match {
     case UIntLiteral(value, IntWidth(width)) =>
       w write s"$width'h${value.toString(16)}"
     case SIntLiteral(value, IntWidth(width)) =>
       val unsignedValue = value + (if (value < 0) BigInt(1) << width.toInt else 0)
       w write s"$width'sh${unsignedValue.toString(16)}"
   }

   def op_stream(doprim: DoPrim): Seq[Any] = {
     def cast_if(e: Expression): Any = {
       doprim.args find (_.tpe match {
         case (_: SIntType) => true
         case (_) => false
       }) match {
         case None => e
         case Some(_) => e.tpe match {
           case (_: SIntType) => Seq("$signed(", e, ")")
           case (_: UIntType) => Seq("$signed({1'b0,", e, "})")
         }
       }
     }
     def cast(e: Expression): Any = doprim.tpe match {
       case (t: UIntType) => e
       case (t: SIntType) => Seq("$signed(",e,")")
     }
     def cast_as(e: Expression): Any = e.tpe match {
       case (t: UIntType) => e
       case (t: SIntType) => Seq("$signed(",e,")")
     }
     def a0: Expression = doprim.args(0)
     def a1: Expression = doprim.args(1)
     def c0: Int = doprim.consts(0).toInt
     def c1: Int = doprim.consts(1).toInt

     def checkArgumentLegality(e: Expression) = e match {
       case _: UIntLiteral | _: SIntLiteral | _: WRef | _: WSubField =>
       case _ => throw new EmitterException(s"Can't emit ${e.getClass.getName} as PrimOp argument")
     }
     doprim.args foreach checkArgumentLegality
     doprim.op match {
       case Add => Seq(cast_if(a0), " + ", cast_if(a1))
       case Addw => Seq(cast_if(a0), " + ", cast_if(a1))
       case Sub => Seq(cast_if(a0), " - ", cast_if(a1))
       case Subw => Seq(cast_if(a0), " - ", cast_if(a1))
       case Mul => Seq(cast_if(a0), " * ", cast_if(a1))
       case Div => Seq(cast_if(a0), " / ", cast_if(a1))
       case Rem => Seq(cast_if(a0), " % ", cast_if(a1))
       case Lt => Seq(cast_if(a0), " < ", cast_if(a1))
       case Leq => Seq(cast_if(a0), " <= ", cast_if(a1))
       case Gt => Seq(cast_if(a0), " > ", cast_if(a1))
       case Geq => Seq(cast_if(a0), " >= ", cast_if(a1))
       case Eq => Seq(cast_if(a0), " == ", cast_if(a1))
       case Neq => Seq(cast_if(a0), " != ", cast_if(a1))
       case Pad =>
         val w = bitWidth(a0.tpe)
         val diff = (c0 - w)
         if (w == 0) Seq(a0)
         else doprim.tpe match {
           // Either sign extend or zero extend.
           // If width == 1, don't extract bit
           case (_: SIntType) if w == 1 => Seq("{", c0, "{", a0, "}}")
           case (_: SIntType) => Seq("{{", diff, "{", a0, "[", w - 1, "]}},", a0, "}")
           case (_) => Seq("{{", diff, "'d0}, ", a0, "}")
         }
       case AsUInt => Seq("$unsigned(", a0, ")")
       case AsSInt => Seq("$signed(", a0, ")")
       case AsClock => Seq("$unsigned(", a0, ")")
       case Dshlw => Seq(cast(a0), " << ", a1)
       case Dshl => Seq(cast(a0), " << ", a1)
       case Dshr => (doprim.tpe) match {
         case (_: SIntType) => Seq(cast(a0)," >>> ", a1)
         case (_) => Seq(cast(a0), " >> ", a1)
       }
       case Shlw => Seq(cast(a0), " << ", c0)
       case Shl => Seq(cast(a0), " << ", c0)
       case Shr if c0 >= bitWidth(a0.tpe) =>
         error("Verilog emitter does not support SHIFT_RIGHT >= arg width")
       case Shr => Seq(a0,"[", bitWidth(a0.tpe) - 1, ":", c0, "]")
       case Neg => Seq("-{", cast(a0), "}")
       case Cvt => a0.tpe match {
         case (_: UIntType) => Seq("{1'b0,", cast(a0), "}")
         case (_: SIntType) => Seq(cast(a0))
       }
       case Not => Seq("~ ", a0)
       case And => Seq(cast_as(a0), " & ", cast_as(a1))
       case Or => Seq(cast_as(a0), " | ", cast_as(a1))
       case Xor => Seq(cast_as(a0), " ^ ", cast_as(a1))
       case Andr => (0 until bitWidth(doprim.tpe).toInt) map (
         Seq(cast(a0), "[", _, "]")) reduce (_ + " & " + _)
       case Orr => (0 until bitWidth(doprim.tpe).toInt) map (
         Seq(cast(a0), "[", _, "]")) reduce (_ + " | " + _)
       case Xorr => (0 until bitWidth(doprim.tpe).toInt) map (
         Seq(cast(a0), "[", _, "]")) reduce (_ + " ^ " + _)
       case Cat => Seq("{", cast(a0), ",", cast(a1), "}")
       // If selecting zeroth bit and single-bit wire, just emit the wire
       case Bits if c0 == 0 && c1 == 0 && bitWidth(a0.tpe) == 1 => Seq(a0)
       case Bits if c0 == c1 => Seq(a0, "[", c0, "]")
       case Bits => Seq(a0, "[", c0, ":", c1, "]")
       case Head =>
         val w = bitWidth(a0.tpe)
         val high = w - 1
         val low = w - c0
         Seq(a0, "[", high, ":", low, "]")
       case Tail =>
         val w = bitWidth(a0.tpe)
         val low = w - c0 - 1
         Seq(a0, "[", low, ":", 0, "]")
     }
   }
   
    def emit_verilog(m: Module)(implicit w: Writer): DefModule = {
      val netlist = LinkedHashMap[WrappedExpression, Expression]()
      val simlist = ArrayBuffer[Statement]()
      val namespace = Namespace(m)
      def build_netlist(s: Statement): Statement = s map build_netlist match {
        case (s: Connect) =>
          netlist(s.loc) = s.expr
          s
        case (s: IsInvalid) =>
          netlist(s.expr) = wref(namespace.newTemp, s.expr.tpe)
          s
        case (s: Conditionally) =>
          simlist += s
          s
        case (s: DefNode) =>
          val e = WRef(s.name, s.value.tpe, NodeKind, MALE)
          netlist(e) = s.value
          s
        case (s) => s
      }
   
      val portdefs = ArrayBuffer[Seq[Any]]()
      val declares = ArrayBuffer[Seq[Any]]()
      val instdeclares = ArrayBuffer[Seq[Any]]()
      val assigns = ArrayBuffer[Seq[Any]]()
      val at_clock = LinkedHashMap[Expression,ArrayBuffer[Seq[Any]]]()
      val initials = ArrayBuffer[Seq[Any]]()
      val simulates = ArrayBuffer[Seq[Any]]()
      def declare (b: String, n: String, t: Type) = t match {
        case (t: VectorType) =>
          declares += Seq(b, " ", t.tpe, " ", n, " [0:", t.size - 1, "];")
        case (t) =>
          declares += Seq(b, " ", t, " ", n,";")
      }
      def assign (e: Expression, value: Expression) {
         assigns += Seq("assign ", e, " = ", value, ";")
      }

      // In simulation, assign garbage under a predicate
      def garbageAssign(e: Expression, syn: Expression, garbageCond: Expression) = {
         assigns += Seq("`ifndef RANDOMIZE_GARBAGE_ASSIGN")
         assigns += Seq("assign ", e, " = ", syn, ";")
         assigns += Seq("`else")
         assigns += Seq("assign ", e, " = ", garbageCond, " ? ", rand_string(syn.tpe), " : ", syn, ";")
         assigns += Seq("`endif")
      }
      def invalidAssign(e: Expression) = {
        assigns += Seq("`ifdef RANDOMIZE_INVALID_ASSIGN")
        assigns += Seq("assign ", e, " = ", rand_string(e.tpe), ";")
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

      def update(e: Expression, value: Expression, clk: Expression, en: Expression) {
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
      def rand_string(t: Type) : Seq[Any] = {
         val nx = namespace.newTemp
         val rand = VRandom(bitWidth(t))
         val tx = SIntType(IntWidth(rand.realWidth))
         declare("reg",nx, tx)
         initials += Seq(wref(nx, tx), " = ", VRandom(bitWidth(t)), ";")
         Seq(nx, "[", bitWidth(t) - 1, ":0]")
      }

      def initialize(e: Expression) = {
        initials += Seq("`ifdef RANDOMIZE_REG_INIT")
        initials += Seq(e, " = ", rand_string(e.tpe), ";")
        initials += Seq("`endif")
      }

      def initialize_mem(s: DefMemory) {
        val index = wref("initvar", s.dataType)
        val rstring = rand_string(s.dataType)
        initials += Seq("`ifdef RANDOMIZE_MEM_INIT")
        initials += Seq("for (initvar = 0; initvar < ", s.depth, "; initvar = initvar+1)")
        initials += Seq(tab, WSubAccess(wref(s.name, s.dataType), index, s.dataType, FEMALE),
                             " = ", rstring,";")
        initials += Seq("`endif")
      }

      def instantiate(n: String,m: String, es: Seq[Expression]) {
         instdeclares += Seq(m, " ", n, " (")
         es.zipWithIndex foreach {case (e, i) =>
           val s = Seq(tab, ".", remove_root(e), "(", LowerTypes.loweredName(e), ")")
           if (i != es.size - 1) instdeclares += Seq(s, ",")
           else instdeclares += s
         }
         instdeclares += Seq(");")
         es foreach { e => 
           declare("wire",LowerTypes.loweredName(e), e.tpe)
           val ex = WRef(LowerTypes.loweredName(e), e.tpe, kind(e), gender(e))
           if (gender(e) == FEMALE) assign(ex,netlist(e))
         }
      }

      def simulate(clk: Expression, en: Expression, s: Seq[Any], cond: Option[String]) {
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

      def stop(ret: Int): Seq[Any] = Seq(if (ret == 0) "$finish;" else "$fatal;")

      def printf(str: StringLit, args: Seq[Expression]): Seq[Any] = {
        val q = '"'.toString
	val strx = s"""$q${VerilogStringLitHandler escape str}$q""" +:
                  (args flatMap (Seq("," , _)))
        Seq("$fwrite(32'h80000002,", strx, ");")
      }

      def delay(e: Expression, n: Int, clk: Expression): Expression = {
        ((0 until n) foldLeft e){(ex, i) =>
          val name = namespace.newTemp
          declare("reg", name, e.tpe)
          val exx = wref(name, e.tpe)
          initialize(exx)
          update(exx, ex, clk, one)
          exx
        }
      }

      def build_ports: Unit = m.ports.zipWithIndex foreach {case (p, i) =>
        p.direction match {
          case Input =>
            portdefs += Seq(p.direction, "  ", p.tpe, " ", p.name)
          case Output =>
            portdefs += Seq(p.direction, " ", p.tpe, " ", p.name)
            val ex = WRef(p.name, p.tpe, PortKind, FEMALE)
            assign(ex, netlist(ex))
        }
      }

      def build_streams(s: Statement): Statement = s map build_streams match {
        case (s: DefWire) => 
          declare("wire",s.name,s.tpe)
          val e = wref(s.name,s.tpe)
          assign(e,netlist(e))
          s
        case (s: DefRegister) =>
          declare("reg", s.name, s.tpe)
          val e = wref(s.name, s.tpe)
          update_and_reset(e, s.clock, s.reset, s.init)
          initialize(e)
          s
        case (s: IsInvalid) =>
          val wref = netlist(s.expr) match { case e: WRef => e }
          declare("reg", wref.name, s.expr.tpe)
          initialize(wref)
          s
        case (s: DefNode) =>
          declare("wire", s.name, s.value.tpe)
          assign(WRef(s.name, s.value.tpe, NodeKind, MALE), s.value)
          s
        case (s: Stop) =>
          val errorString = StringLit(s"${s.ret}\n".getBytes)
          simulate(s.clk, s.en, stop(s.ret), Some("STOP_COND"))
          s
        case (s: Print) =>
          simulate(s.clk, s.en, printf(s.string, s.args), Some("PRINTF_COND"))
          s
        case (s: WDefInstance) =>
          val es = create_exps(WRef(s.name, s.tpe, InstanceKind, MALE))
          instantiate(s.name, s.module, es)
          s
        case (s: DefMemory) =>
          val mem = WRef(s.name, MemPortUtils.memType(s), MemKind, UNKNOWNGENDER)
          def mem_exp (p: String, f: String) = {
            val t1 = field_type(mem.tpe, p)
            val t2 = field_type(t1, f)
            val x = WSubField(mem, p, t1, UNKNOWNGENDER)
            WSubField(x, f, t2, UNKNOWNGENDER)
          }

          declare("reg", s.name, VectorType(s.dataType, s.depth))
          initialize_mem(s)
          for (r <- s.readers) {
            val data = mem_exp(r, "data")
            val addr = mem_exp(r, "addr")
            val en = mem_exp(r, "en")
            // Ports should share an always@posedge, so can't have intermediary wire
            val clk = netlist(mem_exp(r, "clk"))

            declare("wire", LowerTypes.loweredName(data), data.tpe)
            declare("wire", LowerTypes.loweredName(addr), addr.tpe)
            declare("wire", LowerTypes.loweredName(en), en.tpe)

            //; Read port
            assign(addr, netlist(addr)) //;Connects value to m.r.addr
            assign(en, netlist(en))     //;Connects value to m.r.en
            val addr_pipe = delay(addr,s.readLatency-1,clk)
            val en_pipe = if (weq(en,one)) one else delay(en,s.readLatency-1,clk)
            val addrx = if (s.readLatency > 0) {
              val name = namespace.newTemp
              val ref = wref(name, addr.tpe)
              declare("reg", name, addr.tpe)
              initialize(ref)
              update(ref,addr_pipe,clk,en_pipe)
              ref
            } else addr
            val mem_port = WSubAccess(mem,addrx,s.dataType,UNKNOWNGENDER)
            val depthValue = UIntLiteral(s.depth, IntWidth(BigInt(s.depth).bitLength))
            val garbageGuard = DoPrim(Geq, Seq(addrx, depthValue), Seq(), UnknownType)

            if ((s.depth & (s.depth - 1)) == 0)
              assign(data, mem_port)
            else
              garbageAssign(data, mem_port, garbageGuard)
          }
 
          for (w <- s.writers) {
            val data = mem_exp(w, "data")
            val addr = mem_exp(w, "addr")
            val mask = mem_exp(w, "mask")
            val en = mem_exp(w, "en")
            //Ports should share an always@posedge, so can't have intermediary wire
            val clk = netlist(mem_exp(w, "clk"))

            declare("wire", LowerTypes.loweredName(data), data.tpe)
            declare("wire", LowerTypes.loweredName(addr), addr.tpe)
            declare("wire", LowerTypes.loweredName(mask), mask.tpe)
            declare("wire", LowerTypes.loweredName(en), en.tpe)

            //; Write port
            assign(data,netlist(data))
            assign(addr,netlist(addr))
            assign(mask,netlist(mask))
            assign(en,netlist(en))

            val datax = delay(data, s.writeLatency - 1, clk)
            val addrx = delay(addr, s.writeLatency - 1, clk)
            val maskx = delay(mask, s.writeLatency - 1, clk)
            val enx = delay(en, s.writeLatency - 1, clk)
            val mem_port = WSubAccess(mem, addrx, s.dataType, UNKNOWNGENDER)
            update(mem_port, datax, clk, AND(enx, maskx))
          }

          for (rw <- s.readwriters) {
            val wmode = mem_exp(rw, "wmode")
            val rdata = mem_exp(rw, "rdata")
            val wdata = mem_exp(rw, "wdata")
            val wmask = mem_exp(rw, "wmask")
            val addr = mem_exp(rw, "addr")
            val en = mem_exp(rw, "en")
            //Ports should share an always@posedge, so can't have intermediary wire
            val clk = netlist(mem_exp(rw, "clk"))

            declare("wire", LowerTypes.loweredName(wmode), wmode.tpe)
            declare("wire", LowerTypes.loweredName(rdata), rdata.tpe)
            declare("wire", LowerTypes.loweredName(wdata), wdata.tpe)
            declare("wire", LowerTypes.loweredName(wmask), wmask.tpe)
            declare("wire", LowerTypes.loweredName(addr), addr.tpe)
            declare("wire", LowerTypes.loweredName(en), en.tpe)

            //; Assigned to lowered wires of each
            assign(addr, netlist(addr))
            assign(wdata, netlist(wdata))
            assign(addr, netlist(addr))
            assign(wmask, netlist(wmask))
            assign(en, netlist(en))
            assign(wmode, netlist(wmode))

            //; Delay new signals by latency
            val raddrx = if (s.readLatency > 0) delay(addr, s.readLatency - 1, clk) else addr
            val waddrx = delay(addr,s.writeLatency - 1,clk)
            val enx = delay(en,s.writeLatency - 1,clk)
            val wmodex = delay(wmode,s.writeLatency - 1,clk)
            val wdatax = delay(wdata,s.writeLatency - 1,clk)
            val wmaskx = delay(wmask,s.writeLatency - 1,clk)

            val raddrxx = if (s.readLatency > 0) {
              val name = namespace.newTemp
              val ref = wref(name, raddrx.tpe)
              declare("reg", name, raddrx.tpe)
              initialize(ref)
              ref
            } else addr
            val rmem_port = WSubAccess(mem, raddrxx, s.dataType, UNKNOWNGENDER)
            assign(rdata, rmem_port)
            val wmem_port = WSubAccess(mem, waddrx, s.dataType, UNKNOWNGENDER)

            def declare_and_assign(exp: Expression) = {
              val name = namespace.newTemp
              val ref = wref(name, exp.tpe)
              declare("wire", name, exp.tpe)
              assign(ref, exp)
              ref
            }
            val ren = declare_and_assign(NOT(wmodex))
            val wen = declare_and_assign(AND(wmodex, wmaskx))
            if (s.readLatency > 0)
              update(raddrxx, raddrx, clk, AND(enx, ren))
            update(wmem_port, wdatax, clk, AND(enx, wen))
          }
          s
        case s => s
      }
   
      def emit_streams {
        emit(Seq("module ", m.name, "("))
        for ((x, i) <- portdefs.zipWithIndex) {
          if (i != portdefs.size - 1)
            emit(Seq(tab, x, ","))
          else
            emit(Seq(tab, x))
        }
        emit(Seq(");"))

        for (x <- declares) emit(Seq(tab, x))
        for (x <- instdeclares) emit(Seq(tab, x))
        for (x <- assigns) emit(Seq(tab, x))
        if (!initials.isEmpty) {
          emit(Seq("`ifdef RANDOMIZE"))
          emit(Seq("  integer initvar;"))
          emit(Seq("  initial begin"))
          // This enables test benches to set the random values at time 0.001,
          //  then start the simulation later
          // Verilator does not support delay statements, so they are omitted.
          emit(Seq("    `ifndef verilator"))
          emit(Seq("      #0.002 begin end"))
          emit(Seq("    `endif"))
          for (x <- initials) emit(Seq(tab, x))
          emit(Seq("  end"))
          emit(Seq("`endif"))
        }
 
        for (clk_stream <- at_clock if !clk_stream._2.isEmpty) {
          emit(Seq(tab, "always @(posedge ", clk_stream._1, ") begin"))
          for (x <- clk_stream._2) emit(Seq(tab, tab, x))
          emit(Seq(tab, "end"))
        }
        emit(Seq("endmodule"))
      }

      build_netlist(m.body)
      build_ports
      build_streams(m.body)
      emit_streams
      m
   }

   def emit_preamble(implicit w: Writer) {
    emit(Seq(
        "`ifdef RANDOMIZE_GARBAGE_ASSIGN\n",
        "`define RANDOMIZE\n",
        "`endif\n",
        "`ifdef RANDOMIZE_INVALID_ASSIGN\n",
        "`define RANDOMIZE\n",
        "`endif\n",
        "`ifdef RANDOMIZE_REG_INIT\n",
        "`define RANDOMIZE\n",
        "`endif\n",
        "`ifdef RANDOMIZE_MEM_INIT\n",
        "`define RANDOMIZE\n",
        "`endif\n"))
   }

   def run(c: Circuit, w: Writer) = {
     emit_preamble(w)
     c.modules foreach {
       case (m: Module) => emit_verilog(m)(w)
       case (m: ExtModule) =>
     }
   }
}

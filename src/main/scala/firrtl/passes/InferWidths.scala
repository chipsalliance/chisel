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

object InferWidths extends Pass {
  def name = "Infer Widths"
  var mname = ""
  def solve_constraints (l:Seq[WGeq]) : LinkedHashMap[String,Width] = {
    def unique (ls:Seq[Width]) : Seq[Width] = ls.map(w => new WrappedWidth(w)).distinct.map(_.w)
    def make_unique (ls:Seq[WGeq]) : LinkedHashMap[String,Width] = {
      val h = LinkedHashMap[String,Width]()
      for (g <- ls) {
        (g.loc) match {
          case (w:VarWidth) => {
            val n = w.name
            if (h.contains(n)) h(n) = MaxWidth(Seq(g.exp,h(n))) else h(n) = g.exp
          }
          case (w) => w 
        }
      }
      h 
    }
    def simplify (w:Width) : Width = {
      (w map (simplify)) match {
        case (w:MinWidth) => {
          val v = ArrayBuffer[Width]()
          for (wx <- w.args) {
            (wx) match {
              case (wx:MinWidth) => for (x <- wx.args) { v += x }
              case (wx) => v += wx } }
          MinWidth(unique(v)) }
        case (w:MaxWidth) => {
          val v = ArrayBuffer[Width]()
          for (wx <- w.args) {
            (wx) match {
              case (wx:MaxWidth) => for (x <- wx.args) { v += x }
              case (wx) => v += wx } }
          MaxWidth(unique(v)) }
        case (w:PlusWidth) => {
          (w.arg1,w.arg2) match {
            case (w1:IntWidth,w2:IntWidth) => IntWidth(w1.width + w2.width)
            case (w1,w2) => w }}
        case (w:MinusWidth) => {
          (w.arg1,w.arg2) match {
            case (w1:IntWidth,w2:IntWidth) => IntWidth(w1.width - w2.width)
            case (w1,w2) => w }}
        case (w:ExpWidth) => {
          (w.arg1) match {
            case (w1:IntWidth) => IntWidth(BigInt((scala.math.pow(2,w1.width.toDouble) - 1).toLong))
            case (w1) => w }}
        case (w) => w } }
     def substitute (h:LinkedHashMap[String,Width])(w:Width) : Width = {
       //;println-all-debug(["Substituting for [" w "]"])
       val wx = simplify(w)
       //;println-all-debug(["After Simplify: [" wx "]"])
       (simplify(w) map (substitute(h))) match {
         case (w:VarWidth) => {
            //;("matched  println-debugvarwidth!")
            if (h.contains(w.name)) {
               //;println-debug("Contained!")
               //;println-all-debug(["Width: " w])
               //;println-all-debug(["Accessed: " h[name(w)]])
               val t = simplify(substitute(h)(h(w.name)))
               //;val t = h[name(w)]
               //;println-all-debug(["Width after sub: " t])
               h(w.name) = t
               t
            } else w
         }
         case (w) => w
            //;println-all-debug(["not varwidth!" w])
       }
     }
     def b_sub (h:LinkedHashMap[String,Width])(w:Width) : Width = {
       (w map (b_sub(h))) match {
         case (w:VarWidth) => if (h.contains(w.name)) h(w.name) else w
         case (w) => w
       }
     }
     def remove_cycle (n:String)(w:Width) : Width = {
       //;println-all-debug(["Removing cycle for " n " inside " w])
       val wx = (w map (remove_cycle(n))) match {
         case (w:MaxWidth) => MaxWidth(w.args.filter{ w => {
           w match {
             case (w:VarWidth) => !(n equals w.name)
             case (w) => true
           }}})
         case (w:MinusWidth) => {
           w.arg1 match {
             case (v:VarWidth) => if (n == v.name) v else w
             case (v) => w }}
         case (w) => w
       }
       //;println-all-debug(["After removing cycle for " n ", returning " wx])
       wx
     }
     def self_rec (n:String,w:Width) : Boolean = {
       var has = false
       def look (w:Width) : Width = {
         (w map (look)) match {
           case (w:VarWidth) => if (w.name == n) has = true
           case (w) => w }
         w }
       look(w)
       has }
         
     //; Forward solve
     //; Returns a solved list where each constraint undergoes:
     //;  1) Continuous Solving (using triangular solving)
     //;  2) Remove Cycles
     //;  3) Move to solved if not self-recursive
     val u = make_unique(l)
     
     //println("======== UNIQUE CONSTRAINTS ========")
     //for (x <- u) { println(x) }
     //println("====================================")
     
  
     val f = LinkedHashMap[String,Width]()
     val o = ArrayBuffer[String]()
     for (x <- u) {
       //println("==== SOLUTIONS TABLE ====")
       //for (x <- f) println(x)
       //println("=========================")
 
       val (n, e) = (x._1, x._2)
       val e_sub = substitute(f)(e)

       //println("Solving " + n + " => " + e)
       //println("After Substitute: " + n + " => " + e_sub)
       //println("==== SOLUTIONS TABLE (Post Substitute) ====")
       //for (x <- f) println(x)
       //println("=========================")

       val ex = remove_cycle(n)(e_sub)

       //println("After Remove Cycle: " + n + " => " + ex)
       if (!self_rec(n,ex)) {
         //println("Not rec!: " + n + " => " + ex)
         //println("Adding [" + n + "=>" + ex + "] to Solutions Table")
         o += n
         f(n) = ex
       }
     }
  
     //println("Forward Solved Constraints")
     //for (x <- f) println(x)
  
     //; Backwards Solve
     val b = LinkedHashMap[String,Width]()
     for (i <- 0 until o.size) {
       val n = o(o.size - 1 - i)
       /*
       println("SOLVE BACK: [" + n + " => " + f(n) + "]")
       println("==== SOLUTIONS TABLE ====")
       for (x <- b) println(x)
       println("=========================")
       */
       val ex = simplify(b_sub(b)(f(n)))
       /*
       println("BACK RETURN: [" + n + " => " + ex + "]")
       */
       b(n) = ex
       /*
       println("==== SOLUTIONS TABLE (Post backsolve) ====")
       for (x <- b) println(x)
       println("=========================")
       */
     }
     b
  }
     
  def width_BANG (t:Type) : Width = {
    (t) match {
      case (t:UIntType) => t.width
      case (t:SIntType) => t.width
      case ClockType => IntWidth(1)
      case (t) => error("No width!"); IntWidth(-1) } }
  def width_BANG (e:Expression) : Width = width_BANG(e.tpe)

  def reduce_var_widths(c: Circuit, h: LinkedHashMap[String,Width]): Circuit = {
    def evaluate(w: Width): Width = {
      def map2(a: Option[BigInt], b: Option[BigInt], f: (BigInt,BigInt) => BigInt): Option[BigInt] =
        for (a_num <- a; b_num <- b) yield f(a_num, b_num)
      def reduceOptions(l: Seq[Option[BigInt]], f: (BigInt,BigInt) => BigInt): Option[BigInt] =
        l.reduce(map2(_, _, f))

      // This function shouldn't be necessary
      // Added as protection in case a constraint accidentally uses MinWidth/MaxWidth
      // without any actual Widths. This should be elevated to an earlier error
      def forceNonEmpty(in: Seq[Option[BigInt]], default: Option[BigInt]): Seq[Option[BigInt]] =
        if(in.isEmpty) Seq(default)
        else in


      def solve(w: Width): Option[BigInt] = w match {
        case (w: VarWidth) =>
          for{
             v <- h.get(w.name) if !v.isInstanceOf[VarWidth]
             result <- solve(v)
          } yield result
        case (w: MaxWidth) => reduceOptions(forceNonEmpty(w.args.map(solve _), Some(BigInt(0))), max)
        case (w: MinWidth) => reduceOptions(forceNonEmpty(w.args.map(solve _), None), min)
        case (w: PlusWidth) => map2(solve(w.arg1), solve(w.arg2), {_ + _})
        case (w: MinusWidth) => map2(solve(w.arg1), solve(w.arg2), {_ - _})
        case (w: ExpWidth) => map2(Some(BigInt(2)), solve(w.arg1), pow_minus_one)
        case (w: IntWidth) => Some(w.width)
        case (w) => println(w); error("Shouldn't be here"); None;
      }

      val s = solve(w)
      (s) match {
        case Some(s) => IntWidth(s)
        case (s) => w
      }
    }

    def reduce_var_widths_w (w:Width) : Width = {
      //println-all-debug(["REPLACE: " w])
      val wx = evaluate(w)
      //println-all-debug(["WITH: " wx])
      wx
    }
    def reduce_var_widths_s (s: Statement): Statement = {
      def onType(t: Type): Type = t map onType map reduce_var_widths_w
      s map reduce_var_widths_s map onType
    }
 
    val modulesx = c.modules.map{ m => {
      val portsx = m.ports.map{ p => {
        Port(p.info,p.name,p.direction,mapr(reduce_var_widths_w _,p.tpe)) }}
      (m) match {
        case (m:ExtModule) => ExtModule(m.info,m.name,portsx)
        case (m:Module) =>
          mname = m.name
          Module(m.info,m.name,portsx,m.body map reduce_var_widths_s _) }}}
    InferTypes.run(Circuit(c.info,modulesx,c.main))
  }
  
  def run (c:Circuit): Circuit = {
    val v = ArrayBuffer[WGeq]()
    def constrain (w1:Width,w2:Width) : Unit = v += WGeq(w1,w2)
    def get_constraints_t (t1:Type,t2:Type,f:Orientation) : Unit = {
      (t1,t2) match {
        case (t1:UIntType,t2:UIntType) => constrain(t1.width,t2.width)
        case (t1:SIntType,t2:SIntType) => constrain(t1.width,t2.width)
        case (t1:BundleType,t2:BundleType) => {
          (t1.fields,t2.fields).zipped.foreach{ (f1,f2) => {
            get_constraints_t(f1.tpe,f2.tpe,times(f1.flip,f)) }}}
        case (t1:VectorType,t2:VectorType) => get_constraints_t(t1.tpe,t2.tpe,f) }}
    def get_constraints_e (e:Expression) : Expression = {
      (e map (get_constraints_e)) match {
        case (e:Mux) => {
          constrain(width_BANG(e.cond),IntWidth(1))
          constrain(IntWidth(1),width_BANG(e.cond))
          e }
        case (e) => e }}
    def get_constraints (s:Statement) : Statement = {
      (s map (get_constraints_e)) match {
        case (s:Connect) => {
          val n = get_size(s.loc.tpe)
          val ce_loc = create_exps(s.loc)
          val ce_exp = create_exps(s.expr)
          for (i <- 0 until n) {
            val locx = ce_loc(i)
            val expx = ce_exp(i)
            get_flip(s.loc.tpe,i,Default) match {
              case Default => constrain(width_BANG(locx),width_BANG(expx))
              case Flip => constrain(width_BANG(expx),width_BANG(locx)) }}
          s }
        case (s:PartialConnect) => {
          val ls = get_valid_points(s.loc.tpe,s.expr.tpe,Default,Default)
          for (x <- ls) {
             val locx = create_exps(s.loc)(x._1)
             val expx = create_exps(s.expr)(x._2)
             get_flip(s.loc.tpe,x._1,Default) match {
                case Default => constrain(width_BANG(locx),width_BANG(expx))
                case Flip => constrain(width_BANG(expx),width_BANG(locx)) }}
          s }
        case (s:DefRegister) => {
          constrain(width_BANG(s.reset),IntWidth(1))
          constrain(IntWidth(1),width_BANG(s.reset))
          get_constraints_t(s.tpe,s.init.tpe,Default)
          s }
        case (s:Conditionally) => {
          v += WGeq(width_BANG(s.pred),IntWidth(1))
          v += WGeq(IntWidth(1),width_BANG(s.pred))
          s map (get_constraints) }
        case (s) => s map (get_constraints) }}

    for (m <- c.modules) {
      (m) match {
        case (m:Module) => mname = m.name; get_constraints(m.body)
        case (m) => false }}
    //println-debug("======== ALL CONSTRAINTS ========")
    //for x in v do : println-debug(x)
    //println-debug("=================================")
    val h = solve_constraints(v)
    //println-debug("======== SOLVED CONSTRAINTS ========")
    //for x in h do : println-debug(x)
    //println-debug("====================================")
    reduce_var_widths(Circuit(c.info,c.modules,c.main),h)
  }
}

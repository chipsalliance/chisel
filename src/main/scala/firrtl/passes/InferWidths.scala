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
import scala.collection.immutable.ListMap

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._

object InferWidths extends Pass {
  def name = "Infer Widths"
  type ConstraintMap = collection.mutable.LinkedHashMap[String, Width]

  def solve_constraints(l: Seq[WGeq]): ConstraintMap = {
    def unique(ls: Seq[Width]) : Seq[Width] =
      (ls map (new WrappedWidth(_))).distinct map (_.w)
    def make_unique(ls: Seq[WGeq]): ListMap[String,Width] = {
      (ls foldLeft ListMap[String, Width]())((h, g) => g.loc match {
        case w: VarWidth => h get w.name match {
          case None => h + (w.name -> g.exp)
          case Some(p) => h + (w.name -> MaxWidth(Seq(g.exp, p)))
        }
        case _ => h
      })
    }
    def simplify(w: Width): Width = w map simplify match {
      case (w: MinWidth) => MinWidth(unique((w.args foldLeft Seq[Width]()){
        case (res, w: MinWidth) => res ++ w.args
        case (res, w) => res :+ w
      }))
      case (w: MaxWidth) => MaxWidth(unique((w.args foldLeft Seq[Width]()){
        case (res, w: MaxWidth) => res ++ w.args
        case (res, w) => res :+ w
      }))
      case (w: PlusWidth) => (w.arg1, w.arg2) match {
        case (w1: IntWidth, w2 :IntWidth) => IntWidth(w1.width + w2.width)
        case _ => w
      }
      case (w: MinusWidth) => (w.arg1, w.arg2) match {
        case (w1: IntWidth, w2: IntWidth) => IntWidth(w1.width - w2.width)
        case _ => w
      }
      case (w: ExpWidth) => w.arg1 match {
        case (w1: IntWidth) => IntWidth(BigInt((math.pow(2, w1.width.toDouble) - 1).toLong))
        case (w1) => w
      }
      case _ => w
    }

    def substitute(h: ConstraintMap)(w: Width): Width = {
      //;println-all-debug(["Substituting for [" w "]"])
      val wx = simplify(w)
      //;println-all-debug(["After Simplify: [" wx "]"])
      wx map substitute(h) match {
        //;("matched  println-debugvarwidth!")
        case w: VarWidth => h get w.name match {
          case None => w
          case Some(p) =>
            //;println-debug("Contained!")
            //;println-all-debug(["Width: " w])
            //;println-all-debug(["Accessed: " h[name(w)]])
            val t = simplify(substitute(h)(p))
            h(w.name) = t
            t
        }
        case w => w
        //;println-all-debug(["not varwidth!" w])
      }
    }

    def b_sub(h: ConstraintMap)(w: Width): Width = {
      w map b_sub(h) match {
        case w: VarWidth => h getOrElse (w.name, w)
        case w => w
      }
    }

    def remove_cycle(n: String)(w: Width): Width = {
      //;println-all-debug(["Removing cycle for " n " inside " w])
      w map remove_cycle(n) match {
        case w: MaxWidth => MaxWidth(w.args filter {
          case w: VarWidth => !(n equals w.name)
          case w => true
        })
        case w: MinusWidth => w.arg1 match {
          case v: VarWidth if n == v.name => v
          case v => w
        }
        case w => w
      }
      //;println-all-debug(["After removing cycle for " n ", returning " wx])
    }

    def hasVarWidth(n: String)(w: Width): Boolean = {
      var has = false
      def rec(w: Width): Width = {
        w match {
          case w: VarWidth if w.name == n => has = true
          case w =>
        }
        w map rec
      }
      rec(w)
      has
    }
 
    //; Forward solve
    //; Returns a solved list where each constraint undergoes:
    //;  1) Continuous Solving (using triangular solving)
    //;  2) Remove Cycles
    //;  3) Move to solved if not self-recursive
    val u = make_unique(l)
    
    //println("======== UNIQUE CONSTRAINTS ========")
    //for (x <- u) { println(x) }
    //println("====================================")
 
    val f = new ConstraintMap
    val o = ArrayBuffer[String]()
    for ((n, e) <- u) {
      //println("==== SOLUTIONS TABLE ====")
      //for (x <- f) println(x)
      //println("=========================")

      val e_sub = substitute(f)(e)

      //println("Solving " + n + " => " + e)
      //println("After Substitute: " + n + " => " + e_sub)
      //println("==== SOLUTIONS TABLE (Post Substitute) ====")
      //for (x <- f) println(x)
      //println("=========================")

      val ex = remove_cycle(n)(e_sub)

      //println("After Remove Cycle: " + n + " => " + ex)
      if (!hasVarWidth(n)(ex)) {
        //println("Not rec!: " + n + " => " + ex)
        //println("Adding [" + n + "=>" + ex + "] to Solutions Table")
        f(n) = ex
        o += n
      }
    }
 
    //println("Forward Solved Constraints")
    //for (x <- f) println(x)
 
    //; Backwards Solve
    val b = new ConstraintMap
    for (i <- (o.size - 1) to 0 by -1) {
      val n = o(i) // Should visit `o` backward
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
     
  def run (c: Circuit): Circuit = {
    val v = ArrayBuffer[WGeq]()

    def get_constraints_t(t1: Type, t2: Type, f: Orientation): Seq[WGeq] = (t1,t2) match {
      case (t1: UIntType, t2: UIntType) => Seq(WGeq(t1.width, t2.width))
      case (t1: SIntType, t2: SIntType) => Seq(WGeq(t1.width, t2.width))
      case (t1: BundleType, t2: BundleType) =>
        (t1.fields zip t2.fields foldLeft Seq[WGeq]()){case (res, (f1, f2)) =>
          res ++ get_constraints_t(f1.tpe, f2.tpe, times(f1.flip, f))
        }
      case (t1: VectorType, t2: VectorType) => get_constraints_t(t1.tpe, t2.tpe, f)
    }

    def get_constraints_e(e: Expression): Expression = {
      e match {
        case (e: Mux) => v ++= Seq(
          WGeq(getWidth(e.cond), IntWidth(1)),
          WGeq(IntWidth(1), getWidth(e.cond))
        )
        case _ =>
      }
      e map get_constraints_e
    }

    def get_constraints_s(s: Statement): Statement = {
      s match {
        case (s: Connect) =>
          val n = get_size(s.loc.tpe)
          val locs = create_exps(s.loc)
          val exps = create_exps(s.expr)
          v ++= ((locs zip exps).zipWithIndex map {case ((locx, expx), i) =>
            get_flip(s.loc.tpe, i, Default) match {
              case Default => WGeq(getWidth(locx), getWidth(expx))
              case Flip => WGeq(getWidth(expx), getWidth(locx))
            }
          })
        case (s: PartialConnect) =>
          val ls = get_valid_points(s.loc.tpe, s.expr.tpe, Default, Default)
          val locs = create_exps(s.loc)
          val exps = create_exps(s.expr)
          v ++= (ls map {case (x, y) =>
            val locx = locs(x)
            val expx = exps(y)
            get_flip(s.loc.tpe, x, Default) match {
              case Default => WGeq(getWidth(locx), getWidth(expx))
              case Flip => WGeq(getWidth(expx), getWidth(locx))
            }
          })
        case (s:DefRegister) => v ++= (Seq(
           WGeq(getWidth(s.reset), IntWidth(1)),
           WGeq(IntWidth(1), getWidth(s.reset))
        ) ++ get_constraints_t(s.tpe, s.init.tpe, Default))
        case (s:Conditionally) => v ++= Seq(
           WGeq(getWidth(s.pred), IntWidth(1)),
           WGeq(IntWidth(1), getWidth(s.pred))
        )
        case (s: Attach) =>
          v += WGeq(getWidth(s.source), MaxWidth(s.exprs map (e => getWidth(e.tpe))))
        case _ =>
      }
      s map get_constraints_e map get_constraints_s
    }

    c.modules foreach (_ map get_constraints_s)

    //println("======== ALL CONSTRAINTS ========")
    //for(x <- v) println(x)
    //println("=================================")
    val h = solve_constraints(v)
    //println("======== SOLVED CONSTRAINTS ========")
    //for(x <- h) println(x)
    //println("====================================")

    def evaluate(w: Width): Width = {
      def map2(a: Option[BigInt], b: Option[BigInt], f: (BigInt,BigInt) => BigInt): Option[BigInt] =
         for (a_num <- a; b_num <- b) yield f(a_num, b_num)
      def reduceOptions(l: Seq[Option[BigInt]], f: (BigInt,BigInt) => BigInt): Option[BigInt] =
         l.reduce(map2(_, _, f))

      // This function shouldn't be necessary
      // Added as protection in case a constraint accidentally uses MinWidth/MaxWidth
      // without any actual Widths. This should be elevated to an earlier error
      def forceNonEmpty(in: Seq[Option[BigInt]], default: Option[BigInt]): Seq[Option[BigInt]] =
        if (in.isEmpty) Seq(default)
        else in

      def solve(w: Width): Option[BigInt] = w match {
        case (w: VarWidth) =>
          for{
            v <- h.get(w.name) if !v.isInstanceOf[VarWidth]
            result <- solve(v)
          } yield result
        case (w: MaxWidth) => reduceOptions(forceNonEmpty(w.args.map(solve), Some(BigInt(0))), max)
        case (w: MinWidth) => reduceOptions(forceNonEmpty(w.args.map(solve), None), min)
        case (w: PlusWidth) => map2(solve(w.arg1), solve(w.arg2), {_ + _})
        case (w: MinusWidth) => map2(solve(w.arg1), solve(w.arg2), {_ - _})
        case (w: ExpWidth) => map2(Some(BigInt(2)), solve(w.arg1), pow_minus_one)
        case (w: IntWidth) => Some(w.width)
        case (w) => println(w); error("Shouldn't be here"); None;
      }

      solve(w) match {
        case None => w
        case Some(s) => IntWidth(s)
      }
    }

    def reduce_var_widths_w(w: Width): Width = {
      //println-all-debug(["REPLACE: " w])
      evaluate(w)
      //println-all-debug(["WITH: " wx])
    }

    def reduce_var_widths_t(t: Type): Type = {
      t map reduce_var_widths_t map reduce_var_widths_w
    }

    def reduce_var_widths_s(s: Statement): Statement = {
      s map reduce_var_widths_s map reduce_var_widths_t
    }

    def reduce_var_widths_p(p: Port): Port = {
      Port(p.info, p.name, p.direction, reduce_var_widths_t(p.tpe))
    } 
  
    InferTypes.run(c.copy(modules = c.modules map (_
      map reduce_var_widths_p
      map reduce_var_widths_s)))
  }
}

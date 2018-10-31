// See LICENSE for license details.

package firrtl.passes

// Datastructures
import scala.collection.mutable.ArrayBuffer
import scala.collection.immutable.ListMap

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._

object InferWidths extends Pass {
  type ConstraintMap = collection.mutable.LinkedHashMap[String, Width]

  def solve_constraints(l: Seq[WGeq]): ConstraintMap = {
    def unique(ls: Seq[Width]) : Seq[Width] =
      (ls map (new WrappedWidth(_))).distinct map (_.w)
    // Combines constraints on the same VarWidth into the same constraint
    def make_unique(ls: Seq[WGeq]): ListMap[String,Width] = {
      ls.foldLeft(ListMap.empty[String, Width])((acc, wgeq) => wgeq.loc match {
        case VarWidth(name) => acc.get(name) match {
          case None => acc + (name -> wgeq.exp)
          // Avoid constructing massive MaxWidth chains
          case Some(MaxWidth(args)) => acc + (name -> MaxWidth(wgeq.exp +: args))
          case Some(width) => acc + (name -> MaxWidth(Seq(wgeq.exp, width)))
        }
        case _ => acc
      })
    }
    def pullMinMax(w: Width): Width = w map pullMinMax match {
      case PlusWidth(MaxWidth(maxs), IntWidth(i)) => MaxWidth(maxs.map(m => PlusWidth(m, IntWidth(i))))
      case PlusWidth(IntWidth(i), MaxWidth(maxs)) => MaxWidth(maxs.map(m => PlusWidth(m, IntWidth(i))))
      case MinusWidth(MaxWidth(maxs), IntWidth(i)) => MaxWidth(maxs.map(m => MinusWidth(m, IntWidth(i))))
      case MinusWidth(IntWidth(i), MaxWidth(maxs)) => MaxWidth(maxs.map(m => MinusWidth(IntWidth(i), m)))
      case PlusWidth(MinWidth(mins), IntWidth(i)) => MinWidth(mins.map(m => PlusWidth(m, IntWidth(i))))
      case PlusWidth(IntWidth(i), MinWidth(mins)) => MinWidth(mins.map(m => PlusWidth(m, IntWidth(i))))
      case MinusWidth(MinWidth(mins), IntWidth(i)) => MinWidth(mins.map(m => MinusWidth(m, IntWidth(i))))
      case MinusWidth(IntWidth(i), MinWidth(mins)) => MinWidth(mins.map(m => MinusWidth(IntWidth(i), m)))
      case wx => wx
    }
    def collectMinMax(w: Width): Width = w map collectMinMax match {
      case MinWidth(args) => MinWidth(unique(args.foldLeft(List[Width]()) {
        case (res, wxx: MinWidth) => wxx.args ++: res
        case (res, wxx) => wxx +: res
      }))
      case MaxWidth(args) => MaxWidth(unique(args.foldLeft(List[Width]()) {
        case (res, wxx: MaxWidth) => wxx.args ++: res
        case (res, wxx) => wxx +: res
      }))
      case wx => wx
    }
    def mergePlusMinus(w: Width): Width = w map mergePlusMinus match {
      case wx: PlusWidth => (wx.arg1, wx.arg2) match {
        case (w1: IntWidth, w2: IntWidth) => IntWidth(w1.width + w2.width)
        case (PlusWidth(IntWidth(x), w1), IntWidth(y)) =>  PlusWidth(IntWidth(x + y), w1)
        case (PlusWidth(w1, IntWidth(x)), IntWidth(y)) =>  PlusWidth(IntWidth(x + y), w1)
        case (IntWidth(y), PlusWidth(w1, IntWidth(x))) =>  PlusWidth(IntWidth(x + y), w1)
        case (IntWidth(y), PlusWidth(IntWidth(x), w1)) =>  PlusWidth(IntWidth(x + y), w1)
        case (MinusWidth(w1, IntWidth(x)), IntWidth(y)) => PlusWidth(IntWidth(y - x), w1)
        case (IntWidth(y), MinusWidth(w1, IntWidth(x))) => PlusWidth(IntWidth(y - x), w1)
        case _ => wx
      }
      case wx: MinusWidth => (wx.arg1, wx.arg2) match {
        case (w1: IntWidth, w2: IntWidth) => IntWidth(w1.width - w2.width)
        case (PlusWidth(IntWidth(x), w1), IntWidth(y)) =>  PlusWidth(IntWidth(x - y), w1)
        case (PlusWidth(w1, IntWidth(x)), IntWidth(y)) =>  PlusWidth(IntWidth(x - y), w1)
        case (MinusWidth(w1, IntWidth(x)), IntWidth(y)) => PlusWidth(IntWidth(x - y), w1)
        case _ => wx
      }
      case wx: ExpWidth => wx.arg1 match {
        case w1: IntWidth => IntWidth(BigInt((math.pow(2, w1.width.toDouble) - 1).toLong))
        case _ => wx
      }
      case wx => wx
    }
    def removeZeros(w: Width): Width = w map removeZeros match {
      case wx: PlusWidth => (wx.arg1, wx.arg2) match {
        case (w1, IntWidth(x)) if x == 0 => w1
        case (IntWidth(x), w1) if x == 0 => w1
        case _ => wx
      }
      case wx: MinusWidth => (wx.arg1, wx.arg2) match {
        case (w1: IntWidth, w2: IntWidth) => IntWidth(w1.width - w2.width)
        case (w1, IntWidth(x)) if x == 0 => w1
        case _ => wx
      }
      case wx => wx
    }
    def simplify(w: Width): Width = {
      val opts = Seq(
        pullMinMax _,
        collectMinMax _,
        mergePlusMinus _,
        removeZeros _
      )
      opts.foldLeft(w) { (width, opt) => opt(width) }
    }

    def substitute(h: ConstraintMap)(w: Width): Width = {
      //;println-all-debug(["Substituting for [" w "]"])
      val wx = simplify(w)
      //;println-all-debug(["After Simplify: [" wx "]"])
      wx map substitute(h) match {
        //;("matched  println-debugvarwidth!")
        case wxx: VarWidth => h get wxx.name match {
          case None => wxx
          case Some(p) =>
            //;println-debug("Contained!")
            //;println-all-debug(["Width: " wxx])
            //;println-all-debug(["Accessed: " h[name(wxx)]])
            val t = simplify(substitute(h)(p))
            h(wxx.name) = t
            t
        }
        case wxx => wxx
        //;println-all-debug(["not varwidth!" w])
      }
    }

    def b_sub(h: ConstraintMap)(w: Width): Width = {
      w map b_sub(h) match {
        case wx: VarWidth => h getOrElse (wx.name, wx)
        case wx => wx
      }
    }

    def remove_cycle(n: String)(w: Width): Width = {
      //;println-all-debug(["Removing cycle for " n " inside " w])
      w match {
        case wx: MaxWidth => MaxWidth(wx.args filter {
          case wxx: VarWidth => !(n equals wxx.name)
          case MinusWidth(VarWidth(name), IntWidth(i)) if ((i >= 0) && (n == name)) => false
          case _ => true
        })
        case wx: MinusWidth => wx.arg1 match {
          case v: VarWidth if n == v.name => v
          case v => wx
        }
        case wx => wx
      }
      //;println-all-debug(["After removing cycle for " n ", returning " wx])
    }

    def hasVarWidth(n: String)(w: Width): Boolean = {
      var has = false
      def rec(w: Width): Width = {
        w match {
          case wx: VarWidth if wx.name == n => has = true
          case _ =>
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

      val e_sub = simplify(substitute(f)(e))

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

    def get_constraints_t(t1: Type, t2: Type): Seq[WGeq] = (t1,t2) match {
      case (t1: UIntType, t2: UIntType) => Seq(WGeq(t1.width, t2.width))
      case (t1: SIntType, t2: SIntType) => Seq(WGeq(t1.width, t2.width))
      case (ClockType, ClockType) => Nil
      case (FixedType(w1, p1), FixedType(w2, p2)) => Seq(WGeq(w1,w2), WGeq(p1,p2))
      case (AnalogType(w1), AnalogType(w2)) => Seq(WGeq(w1,w2), WGeq(w2,w1))
      case (t1: BundleType, t2: BundleType) =>
        (t1.fields zip t2.fields foldLeft Seq[WGeq]()){case (res, (f1, f2)) =>
          res ++ (f1.flip match {
            case Default => get_constraints_t(f1.tpe, f2.tpe)
            case Flip => get_constraints_t(f2.tpe, f1.tpe)
          })
        }
      case (t1: VectorType, t2: VectorType) => get_constraints_t(t1.tpe, t2.tpe)
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

    def get_constraints_declared_type (t: Type): Type = t match {
      case FixedType(_, p) => 
        v += WGeq(p,IntWidth(0))
        t
      case _ => t map get_constraints_declared_type
    }

    def get_constraints_s(s: Statement): Statement = {
      s map get_constraints_declared_type match {
        case (s: Connect) =>
          val n = get_size(s.loc.tpe)
          val locs = create_exps(s.loc)
          val exps = create_exps(s.expr)
          v ++= locs.zip(exps).flatMap { case (locx, expx) =>
            to_flip(gender(locx)) match {
              case Default => get_constraints_t(locx.tpe, expx.tpe)//WGeq(getWidth(locx), getWidth(expx))
              case Flip => get_constraints_t(expx.tpe, locx.tpe)//WGeq(getWidth(expx), getWidth(locx))
            }
          }
        case (s: PartialConnect) =>
          val ls = get_valid_points(s.loc.tpe, s.expr.tpe, Default, Default)
          val locs = create_exps(s.loc)
          val exps = create_exps(s.expr)
          v ++= (ls flatMap {case (x, y) =>
            val locx = locs(x)
            val expx = exps(y)
            to_flip(gender(locx)) match {
              case Default => get_constraints_t(locx.tpe, expx.tpe)//WGeq(getWidth(locx), getWidth(expx))
              case Flip => get_constraints_t(expx.tpe, locx.tpe)//WGeq(getWidth(expx), getWidth(locx))
            }
          })
        case (s: DefRegister) => v ++= (
           get_constraints_t(s.reset.tpe, UIntType(IntWidth(1))) ++
           get_constraints_t(UIntType(IntWidth(1)), s.reset.tpe) ++ 
           get_constraints_t(s.tpe, s.init.tpe))
        case (s:Conditionally) => v ++= 
           get_constraints_t(s.pred.tpe, UIntType(IntWidth(1))) ++
           get_constraints_t(UIntType(IntWidth(1)), s.pred.tpe)
        case Attach(_, exprs) =>
          // All widths must be equal
          val widths = exprs map (e => getWidth(e.tpe))
          v ++= widths.tail map (WGeq(widths.head, _))
        case _ =>
      }
      s map get_constraints_e map get_constraints_s
    }

    c.modules foreach (_ map get_constraints_s)
    c.modules foreach (_.ports foreach {p => get_constraints_declared_type(p.tpe)})

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
        case wx: VarWidth =>
          for{
            v <- h.get(wx.name) if !v.isInstanceOf[VarWidth]
            result <- solve(v)
          } yield result
        case wx: MaxWidth => reduceOptions(forceNonEmpty(wx.args.map(solve), Some(BigInt(0))), max)
        case wx: MinWidth => reduceOptions(forceNonEmpty(wx.args.map(solve), None), min)
        case wx: PlusWidth => map2(solve(wx.arg1), solve(wx.arg2), {_ + _})
        case wx: MinusWidth => map2(solve(wx.arg1), solve(wx.arg2), {_ - _})
        case wx: ExpWidth => map2(Some(BigInt(2)), solve(wx.arg1), pow_minus_one)
        case wx: IntWidth => Some(wx.width)
        case wx => throwInternalError(s"solve: shouldn't be here - %$wx")
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

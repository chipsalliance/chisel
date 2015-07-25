/*
 Constructing Hardware in a Scala Embedded Language, Copyright (c) 2014, The
 Regents of the University of California, through Lawrence Berkeley National
 Laboratory (subject to receipt of any required approvals from the U.S. Dept.
   of Energy).  All rights reserved.

 If you have questions about your rights to use or distribute this software,
 please contact Berkeley Lab's Technology Transfer Department at  TTD@lbl.gov.

 NOTICE.  This software is owned by the U.S. Department of Energy.  As such,
 the U.S. Government has been granted for itself and others acting on its
 behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
 to reproduce, prepare derivative works, and perform publicly and display
 publicly.  Beginning five (5) years after the date permission to assert
 copyright is obtained from the U.S. Department of Energy, and subject to any
 subsequent five (5) year renewals, the U.S. Government is granted for itself
 and others acting on its behalf a paid-up, nonexclusive, irrevocable,
 worldwide license in the Software to reproduce, prepare derivative works,
 distribute copies to the public, perform publicly and display publicly, and to
 permit others to do so.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 (1) Redistributions of source code must retain the above copyright notice,
 this list of conditions and the following disclaimer.

 (2) Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

 (3) Neither the name of the University of California, Lawrence Berkeley
 National Laboratory, U.S. Dept. of Energy nor the names of its contributors
 may be used to endorse or promote products derived from this software without
 specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 You are under no obligation whatsoever to provide any bug fixes, patches, or
 upgrades to the features, functionality or performance of the source code
 ("Enhancements") to anyone; however, if you choose to make your Enhancements
 available either publicly, or directly to Lawrence Berkeley National
 Laboratory, without imposing a separate written license agreement for such
 Enhancements, then you hereby grant the following license: a  non-exclusive,
 royalty-free perpetual license to install, use, modify, prepare derivative
 works, incorporate into other computer software, distribute, and sublicense
 such enhancements or derivative works thereof, in binary and source code form.

 Authors: J. Bachan, A. Izraelevitz, H. Cook
*/

package Chisel

import scala.collection.immutable.{Seq=>Seq, Iterable=>Iterable}
import scala.{collection=>readonly}
import scala.collection.mutable

// Convention: leading _'s on names means private to the outside world
// but accessible to anything in this file.

abstract trait UsesParameters {
  def params: Parameters
}

class ParameterUndefinedException(field:Any, cause:Throwable=null)
  extends RuntimeException("Parameter " + field + " undefined.", cause)
class KnobUndefinedException(field:Any, cause:Throwable=null)
  extends RuntimeException("Knob " + field + " undefined.", cause)

// Knobs are top level free variables that go into the constraint solver.
final case class Knob[T](name:Any)


class ChiselConfig(
  val topDefinitions: World.TopDefs = { (a,b,c) => {throw new scala.MatchError(a)}},
  val topConstraints: List[ViewSym=>Ex[Boolean]] = List( ex => ExLit[Boolean](true) ),
  val knobValues: Any=>Any = { case x => {throw new scala.MatchError(x)}}
) {
  import Implicits._
  type Constraint = ViewSym=>Ex[Boolean]

  def this(that: ChiselConfig) = this(that.topDefinitions,
                                      that.topConstraints,
                                      that.knobValues)

  def ++(that: ChiselConfig) = {
    new ChiselConfig(this.addDefinitions(that.topDefinitions),
                      this.addConstraints(that.topConstraints),
                      this.addKnobValues(that.knobValues))
  }

  def addDefinitions(that: World.TopDefs): World.TopDefs = {
    (pname,site,here) => {
      try this.topDefinitions(pname, site, here)
      catch {
        case e: scala.MatchError => that(pname, site, here)
      }
    }
  }

  def addConstraints(that: List[Constraint]):List[Constraint] = {
    this.topConstraints ++ that
  }


  def addKnobValues(that: Any=>Any): Any=>Any = { case x =>
    try this.knobValues(x)
    catch {
      case e: scala.MatchError => that(x)
    }
  }

}

object Dump {
  val dump = mutable.Set[Tuple2[Any,Any]]()
  val knobList = mutable.ListBuffer[Any]()
  def apply[T](key:Any,value:T):T = {addToDump(key,value); value}
  def apply[T](knob:Knob[T]):Knob[T] = {knobList += knob.name; knob}
  def addToDump(key:Any,value:Any) = dump += ((key,value))
  def getDump:String = dump.map(_.toString).reduce(_+"\n"+_) + "\n"
}

// objects given to the user in mask functions (site,here,up)
abstract class View {
  // the list of classes in our current path down the heirarchy
  def path: List[Class[_]]

  protected val deftSite: View // when views are queried without a specifying a site this is the default

  // use `this` view's behavior to query for a parameters value as if
  // the original site were `site`
  def apply[T](pname:Any, site:View):T
  def sym[T](pname:Any, site:View):Ex[T]

  // query for a parameters value using the default site
  final def apply[T](pname:Any):T = apply[T](pname, deftSite)
  final def apply[T](field:Field[T]):T = apply[T](field.asInstanceOf[Any], deftSite)

  final def sym[T](pname:Any):Ex[T] = sym[T](pname, deftSite)
  final def sym[T](field:Field[T]):Ex[T] = sym[T](field.asInstanceOf[Any], deftSite)
}

/* Wrap a View to make the application return the symbolic expression,
 * basically a shorthand to save typing '.sym'
 * before:
 *   val v: View
 *   v.sym[Int]("x") // returns Ex[_]
 * now:
 *   val vs = ViewSym(v)
 *   vs[Int]("xs") // Ex[_]
*/
final case class ViewSym(view:View) {
  def apply[T](f:Any):Ex[T] = view.sym[T](f)
  def apply[T](f:Field[T]):Ex[T] = view.sym[T](f)
  def apply[T](f:Any, site:View):Ex[T] = view.sym[T](f, site)
  def apply[T](f:Field[T], site:View):Ex[T] = view.sym[T](f, site)
}


// internal type to represent functions that evaluate parameter values
abstract class _Lookup {
  var path:List[Class[_]] = null

  def apply[T](pname:Any, site:View):Ex[T]

  // build a new Lookup that just defers to this one
  final def push() = {
    val me = this
    new _Lookup {
      this.path = me.path
      def apply[T](pname:Any, site:View) = me.apply(pname, site)
    }
  }
}

// Internal type used as name in all ExVar[T]'s
sealed abstract class _Var[T]

// Variables which are 'free' parameters when seen from the top level.
final case class _VarKnob[T](kname:Any) extends _Var[T] {
  override def toString = kname.toString
}
// Variables whose values are computed by `expr`. The term 'let' comes
// from the idea of 'let' bindings in functional languages i.e.:
final case class _VarLet[T](pname:Any,expr:Ex[T]) extends _Var[T] {
  override def toString = pname.toString + "{" + expr.toString + "}"
}


object World {
  // An alias for the type of function provided by user to describe parameters that
  // reach the top level. The return of this function can be either:
  //   Knob(k): this parameter maps to the constraint variable `k`
  //   Ex: this parameter is computed using the expression
  //   Any(thing else): variable takes a literal value
  type TopDefs = (/*pname:*/Any,/*site:*/View,/*here:*/View) => Any/*Knob[_] | Ex[_] | Any*/
}

// Worlds collect the variable definitions and constraints seen when building hardware.
abstract class World(
    topDefs: World.TopDefs
  ) {

  val _knobs = new mutable.HashSet[Any]
  abstract class _View extends View {
    val look: _Lookup
    def path = look.path

    def apply[T](pname:Any, site:View):T = {
      _eval(look(pname, site).asInstanceOf[Ex[T]])
    }
    def sym[T](pname:Any, site:View):Ex[T] = {
      _bindLet[T](pname,look(pname, site).asInstanceOf[Ex[T]])
    }
  }

  // evaluate an expression against this world
  def _eval[T](e:Ex[T]):T = {
    Ex.eval(e, {
      case v:_VarKnob[_] => {
        _knobs += v.kname
        val e = _knobValue(v.kname)
        if(Dump.knobList.contains(v.kname)) {Dump.addToDump(v.kname,e);e} else e
      }
      case v:_VarLet[_] => _eval(v.expr.asInstanceOf[Ex[T]])
    })
  }

  // create a view whose default site is itself
  def _siteView(look:_Lookup):View = {
    val _look = look
    new _View {
      val look = _look
      val deftSite = this
    }
  }

  // create a View which with a supplied default site
  def _otherView(look:_Lookup, deftSite:View):View = {
    val _look = look
    val _deft = deftSite
    new _View {
      val look = _look
      val deftSite = _deft
    }
  }

  // the top level lookup
  def _topLook():_Lookup = {
    class TopLookup extends _Lookup {
      this.path = Nil

      def apply[T](pname:Any, site:View):Ex[T] = {
        val here = _otherView(this, site)
        (
          try topDefs(pname, site, here)
          catch {
            case e:scala.MatchError => throw new ParameterUndefinedException(pname, e)
          }
        ) match {
          case k:Knob[T] => ExVar[T](_VarKnob[T](k.name))
          case ex:Ex[T] => _bindLet[T](pname,ex)
          case lit => ExLit(lit.asInstanceOf[T])
        }
      }
    }
    new TopLookup
  }

  def _bindLet[T](pname:Any,expr:Ex[T]):Ex[T]

  def _constrain(e:Ex[Boolean]):Unit

  def _knobValue(kname:Any):Any

  def getConstraints:String = ""

  def getKnobs:String = ""
}

// a world responsible for collecting all constraints in the first pass
class Collector(
    topDefs: World.TopDefs,
    knobVal: Any=>Any // maps knob names to default-values
  )
  extends World(topDefs) {

  val _constraints = new mutable.HashSet[Ex[Boolean]]

  def knobs():List[Any] = {
    _knobs.toList
  }

  def constraints():List[Ex[Boolean]] = {
    _constraints.toList
  }

  def _bindLet[T](pname:Any,expr:Ex[T]):Ex[T] = {
    expr match {
      case e:ExVar[T] => expr
      case e:ExLit[T] => expr
      case _ => ExVar[T](_VarLet[T](pname,expr))
    }
  }

  def _constrain(c:Ex[Boolean]) = {
    _constraints += c // add the constraint

    // Also add all equality constraints for all bound variables in the
    // constraint expression and do it recursively for all expressions
    // being bound to.
    var q = List[Ex[_]](c)
    while(!q.isEmpty) {
      val e = q.head  // pop an expression
      q = q.tail
      // walk over the variables in `e`
      for(e <- Ex.unfurl(e)) {
        e match {
          case ExVar(_VarLet(p,e1)) => {
            // form the equality constraint
            val c1 = ExEq[Any](e.asInstanceOf[Ex[Any]], e1.asInstanceOf[Ex[Any]])
            // recurse into the expression if its never been seen before
            if(!_constraints.contains(c1)) {
              _constraints += c1
              q ::= e1 // push
            }
          }
          case _ => {}
        }
      }
    }
  }

  def _knobValue(kname:Any) = {
     try knobVal(kname)
     catch {
       case e:scala.MatchError => throw new KnobUndefinedException(kname, e)
     }
  }

  override def getConstraints:String = if(constraints.isEmpty) "" else constraints.map("( " + _.toString + " )").reduce(_ +"\n" + _) + "\n"

  override def getKnobs:String = if(knobs.isEmpty) "" else {
    knobs.map(_.toString).reduce(_ + "\n" + _) + "\n"
  }
}

// a world instantianted to a specific mapping of knobs to values
class Instance(
    topDefs: World.TopDefs,
    knobVal: Any=>Any
  )
  extends World(topDefs) {

  def _bindLet[T](pname:Any,expr:Ex[T]):Ex[T] = expr
  def _constrain(e:Ex[Boolean]) = {}
  def _knobValue(kname:Any) = {
     try knobVal(kname)
     catch {
       case e:scala.MatchError => throw new KnobUndefinedException(kname, e)
     }
  }
}

object Parameters {
  def root(w:World) = {
    new Parameters(w, w._topLook())
  }
  def empty = Parameters.root(new Collector((a,b,c) => {throw new ParameterUndefinedException(a); a},(a:Any) => {throw new KnobUndefinedException(a); a}))

  // Mask making helpers

  // Lift a regular function into a mask by looking for MatchError's and
  // interpreting those as calls to up
  def makeMask(mask:(Any,View,View,View)=>Any) = {
    (f:Any, site:View, here:View, up:View) => {
      try mask(f,site,here,up)
      catch {case e:MatchError => up.sym[Any](f, site)}
    }
  }

  // Lift a Map to be a mask.
  def makeMask(mask:Map[Any,Any]) = {
    (f:Any, site:View, here:View, up:View) => {
      mask.get(f) match {
        case Some(y) => y
        case None => up.sym[Any](f, site)
      }
    }
  }

  // Lift a PartialFunction to be a mask.
  def makeMask(mask:PartialFunction[Any,Any]) = {
    (f:Any, site:View, here:View, up:View) => {

      if(mask.isDefinedAt(f))
        mask.apply(f)
      else {
        up.sym[Any](f, site)
      }
    }
  }
}

class Field[T]

final class Parameters(
    private val _world: World,
    private val _look: _Lookup
  ) {

  private def _site() = _world._siteView(_look)

  // Create a new Parameters that just defers to this one. This is identical
  // to doing an `alter` but not overriding any values.
  def push():Parameters =
    new Parameters(_world, _look.push())

  // parameter's paths should be immutable but I foresee that not being sufficient
  // when integrated into the chisel Module factory.
  def path = _look.path
  def path_=(x:List[Class[_]]) =
    _look.path = x

  def apply[T](field:Any):T =
    _world._eval(_look(field, _site())).asInstanceOf[T]

  def apply[T](field:Field[T]):T =
    _world._eval(_look(field, _site())).asInstanceOf[T]

  def constrain(gen:ViewSym=>Ex[Boolean]) = {
    val g = gen(new ViewSym(_site()))
    if(!_world._eval(g)) ChiselError.error("Constraint failed: " + g.toString)
    _world._constrain(g)
  }

  private def _alter(mask:(/*field*/Any,/*site*/View,/*here*/View,/*up*/View)=>Any) = {
    class KidLookup extends _Lookup {
      this.path = _look.path

      def apply[T](f:Any, site:View):Ex[T] = {
        val here = _world._otherView(this, site)
        val up = _world._otherView(_look, site)

        mask(f, site, here, up) match {
          case e:Ex[T] => e
          case lit => ExLit(lit.asInstanceOf[T])
        }
      }
    }

    new Parameters(_world, new KidLookup)
  }

  def alter(mask:(/*field*/Any,/*site*/View,/*here*/View,/*up*/View)=>Any) =
    _alter(Parameters.makeMask(mask))

  def alter[T](mask:Map[T,Any]) =
    _alter(Parameters.makeMask(mask.asInstanceOf[Map[Any,Any]]))

  def alterPartial(mask:PartialFunction[Any,Any]) =
    _alter(Parameters.makeMask(mask))
}


/*
 Expression Library
*/
abstract class Ex[T] {
  override def toString = Ex.pretty(this)
}

case class IntEx (expr:Ex[Int]) {
  def === (x:IntEx):Ex[Boolean] = (ExEq[Int](expr,x.expr))
  def +   (x:IntEx):Ex[Int] = ExAdd(expr,x.expr)
  def -   (x:IntEx):Ex[Int] = ExSub(expr,x.expr)
  def *   (x:IntEx):Ex[Int] = ExMul(expr,x.expr)
  def %   (x:IntEx):Ex[Int] = ExMod(expr,x.expr)
  def <   (x:IntEx):Ex[Boolean] = ExLt(expr,x.expr)
  def >   (x:IntEx):Ex[Boolean] = ExGt(expr,x.expr)
  def <=  (x:IntEx):Ex[Boolean] = ExLte(expr,x.expr)
  def >=  (x:IntEx):Ex[Boolean] = ExGte(expr,x.expr)
  def in  (x:List[IntEx]):Ex[Boolean] = {
    val canBound = x.map(_.expr match {
      case e:ExVar[_] => false
      case _ => true
    }).reduce(_ && _)
    if (canBound) {
      val max = x.map(i => Ex.eval(i.expr,(x:Any)=>null)).max
      val min = x.map(i => Ex.eval(i.expr,(x:Any)=>null)).min
      ExAnd(IntEx(expr) in Range(min,max), IntEx(expr) _in x)
    } else {
      IntEx(expr) _in x
    }
  }
  def in  (x:Range):Ex[Boolean] = ExAnd(ExGte(expr,ExLit[Int](x.min)),ExLte(expr,ExLit[Int](x.max)))
  private def _in (x:List[IntEx]):Ex[Boolean] = {
    if (x.isEmpty) ExLit[Boolean](false) else {
      ExOr(IntEx(expr) === x.head,IntEx(expr) _in x.tail)
    }
  }
}

case class BoolEx (expr:Ex[Boolean]) {
  def &&  (x:BoolEx):Ex[Boolean] = ExAnd(expr,x.expr)
  def ||  (x:BoolEx):Ex[Boolean] = ExOr(expr,x.expr)
  def ^   (x:BoolEx):Ex[Boolean] = ExXor(expr,x.expr)
  def === (x:BoolEx):Ex[Boolean] = ExEq[Boolean](expr,x.expr)
  def !== (x:BoolEx):Ex[Boolean] = ExEq[Boolean](expr,x.expr)
}

object Implicits {
  implicit def ExInt_IntEx(i:Ex[Int]):IntEx = IntEx(i)
  implicit def Int_IntEx(i:Int):IntEx = IntEx(ExLit[Int](i))
  implicit def ExBool_BoolEx(b:Ex[Boolean]):BoolEx = BoolEx(b)
  implicit def Bool_IntEx(b:Boolean):BoolEx = BoolEx(ExLit[Boolean](b))

  implicit def ListInt_ListExInt(l:List[Int]):List[IntEx] = l.map((x:Int) => IntEx(ExLit[Int](x)))
  implicit def ListExInt_ListExInt(l:List[Ex[Int]]):List[IntEx] = l.map((x:Ex[Int]) => IntEx(x))
}

final case class ExLit[T](value:T) extends Ex[T]
final case class ExVar[T](name:Any) extends Ex[T]

final case class ExAnd(a:Ex[Boolean], b:Ex[Boolean]) extends Ex[Boolean]
final case class ExOr(a:Ex[Boolean], b:Ex[Boolean]) extends Ex[Boolean]
final case class ExXor(a:Ex[Boolean], b:Ex[Boolean]) extends Ex[Boolean]

final case class ExEq[T](a:Ex[T], b:Ex[T]) extends Ex[Boolean]
final case class ExNeq[T](a:Ex[T], b:Ex[T]) extends Ex[Boolean]

final case class ExLt(a:Ex[Int], b:Ex[Int]) extends Ex[Boolean]
final case class ExLte(a:Ex[Int], b:Ex[Int]) extends Ex[Boolean]
final case class ExGt(a:Ex[Int], b:Ex[Int]) extends Ex[Boolean]
final case class ExGte(a:Ex[Int], b:Ex[Int]) extends Ex[Boolean]
final case class ExAdd(a:Ex[Int], b:Ex[Int]) extends Ex[Int]
final case class ExSub(a:Ex[Int], b:Ex[Int]) extends Ex[Int]
final case class ExMul(a:Ex[Int], b:Ex[Int]) extends Ex[Int]
final case class ExMod(a:Ex[Int], b:Ex[Int]) extends Ex[Int]

object Ex {
  // evaluate an expression given a context that maps variable names to values
  def eval[T](e:Ex[T], ctx:Any=>Any):T = e match {
    case ExLit(v) => v.asInstanceOf[T]
    case ExVar(nm) => ctx(nm).asInstanceOf[T]
    case ExAnd(a,b) => eval(a,ctx) && eval(b,ctx)
    case ExOr(a,b) => eval(a,ctx) || eval(b,ctx)
    case ExXor(a,b) => eval(a,ctx) ^ eval(b,ctx)
    case e:ExEq[u] => eval(e.a,ctx) == eval(e.b,ctx)
    case e:ExNeq[u] => eval(e.a,ctx) != eval(e.b,ctx)
    case ExLt(a,b) => eval(a,ctx) < eval(b,ctx)
    case ExLte(a,b) => eval(a,ctx) <= eval(b,ctx)
    case ExGt(a,b) => eval(a,ctx) > eval(b,ctx)
    case ExGte(a,b) => eval(a,ctx) >= eval(b,ctx)
    case ExAdd(a,b) => eval(a,ctx) + eval(b,ctx)
    case ExSub(a,b) => eval(a,ctx) - eval(b,ctx)
    case ExMul(a,b) => eval(a,ctx) * eval(b,ctx)
    case ExMod(a,b) => eval(a,ctx) % eval(b,ctx)
  }

  // get shallow list of subexpressions
  def subExs(e:Ex[_]):List[Ex[_]] = e match {
    case ExLit(_) => Nil
    case ExVar(_) => Nil
    case ExAnd(a,b) => List(a,b)
    case ExOr(a,b) => List(a,b)
    case ExXor(a,b) => List(a,b)
    case ExEq(a,b) => List(a,b)
    case ExNeq(a,b) => List(a,b)
    case ExLt(a,b) => List(a,b)
    case ExLte(a,b) => List(a,b)
    case ExGt(a,b) => List(a,b)
    case ExGte(a,b) => List(a,b)
    case ExAdd(a,b) => List(a,b)
    case ExSub(a,b) => List(a,b)
    case ExMul(a,b) => List(a,b)
    case ExMod(a,b) => List(a,b)
  }

  // get all subexpressions including the expression given
  def unfurl(e:Ex[_]):List[Ex[_]] =
    e :: (subExs(e) flatMap unfurl)

  // pretty-print expression
  def pretty(e:Ex[_]):String = {
    // precedence rank for deciding where to put parentheses
    def rank(e:Ex[_]):Int = e match {
      case e:ExAnd => 40
      case e:ExOr => 50
      case e:ExXor => 50
      case e:ExEq[_] => 30
      case e:ExNeq[_] => 30
      case e:ExLt => 30
      case e:ExLte => 30
      case e:ExGt => 30
      case e:ExGte => 30
      case e:ExAdd => 20
      case e:ExSub => 20
      case e:ExMul => 20
      case e:ExMod => 20
      case e:ExLit[_] => 0
      case e:ExVar[_] => 0
    }

    val r = rank(e)

    def term(t:Ex[_]):String = {
      val rt = rank(t)
      //if(rt >= r)
        "( " + t.toString + " )"
      //else
        //t.toString
    }

    import Implicits._
    e match {
      case ExLit(v) => v.toString
      case e:ExVar[_]=> "$"+e.name
      case ExAnd(a,b) => term(a)+" && "+term(b)
      case ExOr(a,b) => term(a)+" || "+term(b)
      case ExXor(a,b) => term(a)+" ^ "+term(b)
      case ExEq(a,b) => term(a)+" = "+term(b)
      case ExNeq(a,b) => term(a)+" != "+term(b)
      case ExLt(a,b) => term(a)+" < "+term(b)
      case ExLte(a,b) => term(a)+" <= "+term(b)
      case ExGt(a,b) => term(a)+" > "+term(b)
      case ExGte(a,b) => term(a)+" >= "+term(b)
      case ExAdd(a,b) => term(a)+" + "+term(b)
      case ExSub(a,b) => term(a)+" - "+term(b)
      case ExMul(a,b) => term(a)+" * "+term(b)
      case ExMod(a,b) => term(a)+" % "+term(b)
    }
  }
}

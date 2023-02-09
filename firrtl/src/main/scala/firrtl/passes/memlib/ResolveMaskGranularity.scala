// SPDX-License-Identifier: Apache-2.0

package firrtl.passes
package memlib

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import WrappedExpression.weq

object AnalysisUtils {
  type Connects = collection.mutable.HashMap[String, Expression]

  /** Builds a map from named component to assigned value
    * Named components are serialized LHS of connections, nodes, invalids
    */
  def getConnects(m: DefModule): Connects = {
    def getConnects(connects: Connects)(s: Statement): Statement = {
      s match {
        case Connect(_, loc, expr) =>
          connects(loc.serialize) = expr
        case DefNode(_, name, value) =>
          connects(name) = value
        case IsInvalid(_, value) =>
          connects(value.serialize) = WInvalid
        case _ => // do nothing
      }
      s.map(getConnects(connects))
    }
    val connects = new Connects
    m.map(getConnects(connects))
    connects
  }

  /** Find a connection LHS's origin from a module's list of node-to-node connections
    *   regardless of whether constant propagation has been run.
    * Will search past trivial primop/mux's which do not affect its origin.
    * Limitations:
    *   - Only works in a module (stops @ module inputs)
    *   - Only does trivial primop/mux's (is not complete)
    * TODO(shunshou): implement more equivalence cases (i.e. a + 0 = a)
    */
  def getOrigin(connects: Connects, s: String): Expression =
    getOrigin(connects)(WRef(s, UnknownType, ExpKind, UnknownFlow))
  def getOrigin(connects: Connects)(e: Expression): Expression = e match {
    case Mux(cond, tv, fv, _) =>
      val fvOrigin = getOrigin(connects)(fv)
      val tvOrigin = getOrigin(connects)(tv)
      val condOrigin = getOrigin(connects)(cond)
      if (weq(tvOrigin, one) && weq(fvOrigin, zero)) condOrigin
      else if (weq(condOrigin, one)) tvOrigin
      else if (weq(condOrigin, zero)) fvOrigin
      else if (weq(tvOrigin, fvOrigin)) tvOrigin
      else if (weq(fvOrigin, zero) && weq(condOrigin, tvOrigin)) condOrigin
      else e
    case DoPrim(PrimOps.Or, args, consts, tpe) if args.exists(weq(_, one))   => one
    case DoPrim(PrimOps.And, args, consts, tpe) if args.exists(weq(_, zero)) => zero
    case DoPrim(PrimOps.Bits, args, Seq(msb, lsb), tpe) =>
      val extractionWidth = (msb - lsb) + 1
      val nodeWidth = bitWidth(args.head.tpe)
      // if you're extracting the full bitwidth, then keep searching for origin
      if (nodeWidth == extractionWidth) getOrigin(connects)(args.head) else e
    case DoPrim((PrimOps.AsUInt | PrimOps.AsSInt | PrimOps.AsClock), args, _, _) =>
      getOrigin(connects)(args.head)
    // It is a correct optimization to treat ValidIf as a connection
    case ValidIf(cond, value, _) => getOrigin(connects)(value)
    // note: this should stop on a reg, but will stack overflow for combinational loops (not allowed)
    case _: WRef | _: WSubField | _: WSubIndex | _: WSubAccess if kind(e) != RegKind =>
      connects.get(e.serialize) match {
        case Some(ex) => getOrigin(connects)(ex)
        case None     => e
      }
    case _ => e
  }
}

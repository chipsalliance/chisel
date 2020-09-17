// SPDX-License-Identifier: Apache-2.0
// Author: Kevin Laeufer <laeufer@cs.berkeley.edu>

package firrtl.backends.experimental.smt

/** Similar to the mapExpr and foreachExpr methods of the firrtl ir nodes, but external to the case classes */
private object SMTExprVisitor {
  type ArrayFun = ArrayExpr => ArrayExpr
  type BVFun = BVExpr => BVExpr

  def map[T <: SMTExpr](bv: BVFun, ar: ArrayFun)(e: T): T = e match {
    case b: BVExpr    => map(b, bv, ar).asInstanceOf[T]
    case a: ArrayExpr => map(a, bv, ar).asInstanceOf[T]
  }
  def map[T <: SMTExpr](f: SMTExpr => SMTExpr)(e: T): T =
    map(b => f(b).asInstanceOf[BVExpr], a => f(a).asInstanceOf[ArrayExpr])(e)

  private def map(e: BVExpr, bv: BVFun, ar: ArrayFun): BVExpr = e match {
    // nullary
    case old: BVLiteral => bv(old)
    case old: BVSymbol  => bv(old)
    case old: BVRawExpr => bv(old)
    // unary
    case old @ BVExtend(e, by, signed) => val n = map(e, bv, ar); bv(if (n.eq(e)) old else BVExtend(n, by, signed))
    case old @ BVSlice(e, hi, lo)      => val n = map(e, bv, ar); bv(if (n.eq(e)) old else BVSlice(n, hi, lo))
    case old @ BVNot(e)                => val n = map(e, bv, ar); bv(if (n.eq(e)) old else BVNot(n))
    case old @ BVNegate(e)             => val n = map(e, bv, ar); bv(if (n.eq(e)) old else BVNegate(n))
    case old @ BVReduceAnd(e)          => val n = map(e, bv, ar); bv(if (n.eq(e)) old else BVReduceAnd(n))
    case old @ BVReduceOr(e)           => val n = map(e, bv, ar); bv(if (n.eq(e)) old else BVReduceOr(n))
    case old @ BVReduceXor(e) => val n = map(e, bv, ar); bv(if (n.eq(e)) old else BVReduceXor(n))
    // binary
    case old @ BVImplies(a, b) =>
      val (nA, nB) = (map(a, bv, ar), map(b, bv, ar))
      bv(if (nA.eq(a) && nB.eq(b)) old else BVImplies(nA, nB))
    case old @ BVEqual(a, b) =>
      val (nA, nB) = (map(a, bv, ar), map(b, bv, ar))
      bv(if (nA.eq(a) && nB.eq(b)) old else BVEqual(nA, nB))
    case old @ ArrayEqual(a, b) =>
      val (nA, nB) = (map(a, bv, ar), map(b, bv, ar))
      bv(if (nA.eq(a) && nB.eq(b)) old else ArrayEqual(nA, nB))
    case old @ BVComparison(op, a, b, signed) =>
      val (nA, nB) = (map(a, bv, ar), map(b, bv, ar))
      bv(if (nA.eq(a) && nB.eq(b)) old else BVComparison(op, nA, nB, signed))
    case old @ BVOp(op, a, b) =>
      val (nA, nB) = (map(a, bv, ar), map(b, bv, ar))
      bv(if (nA.eq(a) && nB.eq(b)) old else BVOp(op, nA, nB))
    case old @ BVConcat(a, b) =>
      val (nA, nB) = (map(a, bv, ar), map(b, bv, ar))
      bv(if (nA.eq(a) && nB.eq(b)) old else BVConcat(nA, nB))
    case old @ ArrayRead(a, b) =>
      val (nA, nB) = (map(a, bv, ar), map(b, bv, ar))
      bv(if (nA.eq(a) && nB.eq(b)) old else ArrayRead(nA, nB))
    // ternary
    case old @ BVIte(a, b, c) =>
      val (nA, nB, nC) = (map(a, bv, ar), map(b, bv, ar), map(c, bv, ar))
      bv(if (nA.eq(a) && nB.eq(b) && nC.eq(c)) old else BVIte(nA, nB, nC))
  }

  private def map(e: ArrayExpr, bv: BVFun, ar: ArrayFun): ArrayExpr = e match {
    case old: ArrayRawExpr => ar(old)
    case old: ArraySymbol  => ar(old)
    case old @ ArrayConstant(e, indexWidth) =>
      val n = map(e, bv, ar); ar(if (n.eq(e)) old else ArrayConstant(n, indexWidth))
    case old @ ArrayStore(a, b, c) =>
      val (nA, nB, nC) = (map(a, bv, ar), map(b, bv, ar), map(c, bv, ar))
      ar(if (nA.eq(a) && nB.eq(b) && nC.eq(c)) old else ArrayStore(nA, nB, nC))
    case old @ ArrayIte(a, b, c) =>
      val (nA, nB, nC) = (map(a, bv, ar), map(b, bv, ar), map(c, bv, ar))
      ar(if (nA.eq(a) && nB.eq(b) && nC.eq(c)) old else ArrayIte(nA, nB, nC))
  }

}

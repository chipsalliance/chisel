// SPDX-License-Identifier: Apache-2.0

package firrtl

import firrtl.ir._
import firrtl.PrimOps._
import firrtl.traversals.Foreachers._

import scala.collection.mutable

import _root_.logger.LazyLogging

object getWidth {
  def apply(t: Type): Width = t match {
    case t: GroundType => t.width
    case _ => Utils.error(s"No width: $t")
  }
  def apply(e: Expression): Width = apply(e.tpe)
}

object Utils extends LazyLogging {

  /** Unwind the causal chain until we hit the initial exception (which may be the first).
    *
    * @param maybeException - possible exception triggering the error,
    * @param first - true if we want the first (eldest) exception in the chain,
    * @return first or last Throwable in the chain.
    */
  def getThrowable(maybeException: Option[Throwable], first: Boolean): Throwable = {
    maybeException match {
      case Some(e: Throwable) => {
        val t = e.getCause
        if (t != null) {
          if (first) {
            getThrowable(Some(t), first)
          } else {
            t
          }
        } else {
          e
        }
      }
      case None | null => null
    }
  }

  /** Throw an internal error, possibly due to an exception.
    *
    * @param message - possible string to emit,
    * @param exception - possible exception triggering the error.
    */
  def throwInternalError(message: String = "", exception: Option[Throwable] = None) = {
    // We'll get the first exception in the chain, keeping it intact.
    val first = true
    val throwable = getThrowable(exception, true)
    val string = if (message.nonEmpty) message + "\n" else message
    error(
      "Internal Error! %sPlease file an issue at https://github.com/ucb-bar/firrtl/issues".format(string),
      throwable
    )
  }

  def time[R](block: => R): (Double, R) = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    val timeMillis = (t1 - t0) / 1000000.0
    (timeMillis, result)
  }

  def getUIntWidth(u: BigInt):     Int = u.bitLength
  def dec2string(v:   BigDecimal): String = v.underlying().stripTrailingZeros().toPlainString
  def trim(v:         BigDecimal): BigDecimal = BigDecimal(dec2string(v))
  val BoolType = UIntType(IntWidth(1))
  val one = UIntLiteral(1)
  val zero = UIntLiteral(0)

  def sub_type(v: Type): Type = v match {
    case vx: VectorType => vx.tpe
    case vx => UnknownType
  }
  def field_type(v: Type, s: String): Type = v match {
    case vx: BundleType =>
      vx.fields.find(_.name == s) match {
        case Some(f) => f.tpe
        case None    => UnknownType
      }
    case vx => UnknownType
  }

// =================================
  def error(str: String, cause: Throwable = null) = throw new FirrtlInternalException(str, cause)

// =========== ACCESSORS =========
  /** Similar to Seq.groupBy except that it preserves ordering of elements within each group */
  def groupByIntoSeq[A, K](xs: Iterable[A])(f: A => K): Seq[(K, Seq[A])] = {
    val map = mutable.LinkedHashMap.empty[K, mutable.ListBuffer[A]]
    for (x <- xs) {
      val key = f(x)
      val l = map.getOrElseUpdate(key, mutable.ListBuffer.empty[A])
      l += x
    }
    map.view.map({ case (k, vs) => k -> vs.toList }).toList
  }

  // For a given module, returns a Seq of all instantiated modules inside of it
  private[firrtl] def collectInstantiatedModules(mod: Module, map: Map[String, DefModule]): Seq[DefModule] = {
    // Use list instead of set to maintain order
    val modules = mutable.ArrayBuffer.empty[DefModule]
    def onStmt(stmt: Statement): Unit = stmt match {
      case DefInstance(_, _, name, _) => modules += map(name)
      case other                      => other.foreach(onStmt)
    }
    onStmt(mod.body)
    modules.distinct.toSeq
  }

  /** Checks if two circuits are equal regardless of their ordering of module definitions */
  def orderAgnosticEquality(a: Circuit, b: Circuit): Boolean =
    a.copy(modules = a.modules.sortBy(_.name)) == b.copy(modules = b.modules.sortBy(_.name))

  /** Combines several separate circuit modules (typically emitted by -e or -p compiler options) into a single circuit */
  def combine(circuits: Seq[Circuit]): Circuit = {
    def dedup(modules: Seq[DefModule]): Seq[Either[Module, DefModule]] = {
      // Left means module with no ExtModules, Right means child modules or lone ExtModules
      val module: Option[Module] = {
        val found: Seq[Module] = modules.collect { case m: Module => m }
        assert(
          found.size <= 1,
          s"Module definitions should have unique names, found ${found.size} definitions named ${found.head.name}"
        )
        found.headOption
      }
      val extModules: Seq[ExtModule] = modules.collect { case e: ExtModule => e }.distinct

      // If the module is a lone module (no extmodule references in any other file)
      if (extModules.isEmpty && !module.isEmpty)
        Seq(Left(module.get))
      // If a module has extmodules, but no other file contains the implementation
      else if (!extModules.isEmpty && module.isEmpty)
        extModules.map(Right(_))
      // Otherwise there is a module implementation with extmodule references
      else
        Seq(Right(module.get))
    }

    // 1. Combine modules
    val grouped: Seq[(String, Seq[DefModule])] = groupByIntoSeq(circuits.flatMap(_.modules))({
      case mod: Module    => mod.name
      case ext: ExtModule => ext.defname
    })
    val deduped: Iterable[Either[Module, DefModule]] = grouped.flatMap { case (_, insts) => dedup(insts) }

    // 2. Determine top
    val top = {
      val found = deduped.collect { case Left(m) => m }
      assert(found.size == 1, s"There should only be 1 top module, got: ${found.map(_.name).mkString(", ")}")
      found.head
    }
    val res = deduped.collect { case Right(m) => m }
    ir.Circuit(NoInfo, top +: res.toSeq, top.name)
  }

}

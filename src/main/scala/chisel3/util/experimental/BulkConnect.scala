package chisel3.util.experimental

import chisel3._
import chisel3.experimental.DataMirror

import scala.collection.mutable

trait CustomBulkAssignable { this: Data =>
  def assignL(that: CustomBulkAssignable): Unit
  def assignR(that: CustomBulkAssignable): Unit
}


/** provides operators useful for working with bidirectional [[Bundle]]s
  *
  * In terms of [[Flipped]] with a producer 'p' and 'consumer' c:
  * `c :<= p` means drive all unflipped fields of 'c' from 'p' (e.g.: c.valid := p.valid)
  * `c :=> p` means drive all flipped fields of 'p' from 'c' (e.g.: `p.ready := c.ready`)
  * `c :<> p` do both of the above
  * `p :<> c` do both of the above, but you'll probably get a Flow error later.
  *
  * In [[chisel3.Data]] API:
  * `c := p` only works if there are no directions on fields.
  * `c <> p` only works if one of those is an [[chisel3.experimental.IO]] (not a [[Wire]]).
  *
  * Compared with [[chisel3.Data]] operators:
  * `c <> p` is an 'actual-direction'-inferred 'c :<> p' or 'p :<> c'
  * `c := p` is equivalent to 'c :<= p' + 'p :=> c'.
  *          In other words, drive ALL fields of 'c' from 'p' regardless of their direction.
  *
  * Contrast this with 'c :<> p' which will connect a ready-valid producer
  * 'p' to a consumer 'c'.
  * If you flip this to 'p :<> c', it works the way you would expect (flipping the role of producer/consumer).
  * This is how compatibility mode and FIRRTL work.
  * Some find that ':<>' has superior readability (even if the direction can be inferred from an IO),
  * because it clearly states the intended producer/consumer relationship.
  * You will get an appropriate error if you connected it the wrong way
  * (usually because you got the IO direction wrong) instead of silently succeeding.
  *
  * What if you want to connect all of the signals (e.g. ready/valid/bits)
  * from producer 'p' to a monitor 'm'?
  * For example in order to tap the connection to monitor traffic on an existing connection.
  * In that case you can do 'm :<= p' and 'p :=> m'.
  */
object BulkConnect {
  implicit class DataToBulkConnect[T <: Data](val x: T) {
    /** Assign all output fields of x from y; note that the actual direction of x is irrelevant */
    def :<= (y: T): Unit = assignL(x, y)
    /** Assign all input fields of y from x; note that the actual direction of y is irrelevant */
    def :=> (y: T): Unit = assignR(x, y)
    /** Bulk connect which will work between two [[Wire]]s (in addition to between [[chisel3.experimental.IO]]s) */
    def :<> (y: T): Unit = {
      assignL(x, y)
      assignR(x, y)
    }
  }

  // Used by :<= for child elements to switch directionality
  private[chisel3] def descendL(x: Data, y: Data): Unit = {
    DataMirror.specifiedDirectionOf(x) match {
      case SpecifiedDirection.Unspecified => assignL(x, y)
      case SpecifiedDirection.Output      => assignL(x, y); assignR(y, x)
      case SpecifiedDirection.Input       => ()
      case SpecifiedDirection.Flip        => assignR(y, x)
    }
  }

  // Used by :=> for child elements to switch directionality
  private[chisel3] def descendR(x: Data, y: Data): Unit = {
    DataMirror.specifiedDirectionOf(y) match {
      case SpecifiedDirection.Unspecified => assignR(x, y)
      case SpecifiedDirection.Output      => ()
      case SpecifiedDirection.Input       => assignL(y, x); assignR(x, y)
      case SpecifiedDirection.Flip        => assignL(y, x)
    }
  }

  // The default implementation of 'x :<= y'
  // Assign all output fields of x from y
  private[chisel3] def assignL(x: Data, y: Data): Unit = {
    (x, y) match {
      case (cx: CustomBulkAssignable, cy: CustomBulkAssignable) => cx.assignL(cy)
      case (vx: Vec[_], vy: Vec[_]) =>
        require (vx.size == vy.size, s"Assignment between vectors of unequal length (${vx.size} != ${vy.size})")
        (vx zip vy) foreach { case (ex, ey) => descendL(ex, ey) }
      case (rx: Record, ry: Record) =>
        val hy = mutable.HashMap(ry.elements.toList:_*)
        rx.elements.foreach { case (key, vx) =>
          require (hy.contains(key), s"Attempt to assign $x :<= $y, where RHS is missing field $key")
          descendL(vx, hy(key))
        }
        hy --= rx.elements.keys
        require (hy.isEmpty, s"Attempt to assign $x :<= $y, where RHS has excess field ${hy.last._1}")
      case (vx: Vec[_], DontCare) => vx.foreach(ex => descendL(ex, DontCare))
      case (rx: Record, DontCare) => rx.elements.foreach { case (_, dx) => descendL(dx, DontCare) }
      case _ => x := y // assign leaf fields (UInt/etc)
    }
  }

  // The default implementation of 'x :=> y'
  // Assign all input fields of y from x
  private[chisel3] def assignR(x: Data, y: Data): Unit = {
    (x, y) match {
      case (cx: CustomBulkAssignable, cy: CustomBulkAssignable) => cx.assignR(cy)
      case (vx: Vec[_], vy: Vec[_]) =>
        require (vx.size == vy.size, s"Assignment between vectors of unequal length (${vx.size} != ${vy.size})")
        (vx zip vy) foreach { case (ex, ey) => descendR(ex, ey) }
      case (rx: Record, ry: Record) =>
        val hx = mutable.HashMap(rx.elements.toList:_*)
        ry.elements.foreach { case (key, vy) =>
          require (hx.contains(key), s"Attempt to assign $x :=> $y, where RHS has excess field $key")
          descendR(hx(key), vy)
        }
        hx --= ry.elements.keys
        require (hx.isEmpty, s"Attempt to assign $x :=> $y, where RHS is missing field ${hx.last._1}")
      case (DontCare, vy: Vec[_]) => vy.foreach(ey => descendR(DontCare, ey))
      case (DontCare, ry: Record) => ry.elements.foreach { case (_, dy) => descendR(DontCare, dy) }
      case _ =>  // no-op for leaf fields
    }
  }
}

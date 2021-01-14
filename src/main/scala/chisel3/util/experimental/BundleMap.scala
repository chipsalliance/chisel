package chisel3.util.experimental


import chisel3._
import chisel3.experimental.DataMirror
import chisel3.util.experimental.BulkConnect.DataToBulkConnect

import scala.collection.immutable.ListMap
import scala.collection.mutable

/* BundleMaps include IOs for every BundleField they are constructed with.
 * A given BundleField in a BundleMap is accessed by a BundleKey.
 *
 * So, for example:
 *   val myBundleMap = BundleMap(Seq(MyBundleDataField(width = 8), ...))
 *   myBundleMap(MyBundleData) := 7.U
 *
 * case object MyBundleData extends DataKey[UInt]("data") // "data" is the name of the IO used in BundleMaps
 * case class MyBundleDataField(width: Int) extends BundleField(MyBundleData) {
 *   def data = Output(UInt(width.W))
 *   def default(x: UInt) { x := 0.U }
 * }
 * OR:
 * case class MyBundleDataField(width: Int) extends SimpleBundleField(MyBundleData)(Output(UInt(width.W)), 0.U)
 */

sealed trait BundleFieldBase {
  def key: BundleKeyBase
  def data: Data // the field's chisel type with a direction
  def setDataDefault(x: Data): Unit

  // Overload this if there is a way to unify differently parameterized cases of a field
  // (For example, by selecting the widest width)
  def unify(that: BundleFieldBase): BundleFieldBase = {
    require (this == that, s"Attempted to unify two BundleMaps with conflicting fields: $this and $that")
    that
  }
}

/* Always extends BundleField with a case class.
 * This will ensure that there is an appropriate equals() operator to detect name conflicts.
 */
abstract class BundleField[T <: Data](val key: BundleKey[T]) extends BundleFieldBase
{
  def data: T
  def default(x: T): Unit
  def setDataDefault(x: Data): Unit = default(x.asInstanceOf[T])
}

abstract class SimpleBundleField[T <: Data](key: BundleKey[T])(typeT: => T, defaultT: => T) extends BundleField(key)
{
  def data: T = typeT
  def default(x: T): Unit = { x := defaultT }
}

object BundleField {
  /* Consider an arbiter that receives two request streams A and B and combines them to C.
   * The output stream C should have the union of all keys from A and B.
   * When a key from A and B have the same name:
   *  - it is an error if they are not equal.
   *  - the union contains only one copy.
   */
  def union(fields: Seq[BundleFieldBase]): Seq[BundleFieldBase] =
    fields.groupBy(_.key.name).map(_._2.reduce(_ unify _)).toList
  /* There is no point in carrying an extra field if the other end does not use it.
   */
  def accept(fields: Seq[BundleFieldBase], keys: Seq[BundleKeyBase]): Seq[BundleFieldBase] = {
    def hk = mutable.HashMap(keys.map(k => (k.name, k)):_*)
    fields.filter(f => hk.get(f.key.name).contains(f.key))
  }
}

sealed trait BundleKeyBase {
  def name: String
}

sealed class BundleKey[T <: Data](val name: String) extends BundleKeyBase


// If you extend this class, you must either redefine cloneType or have a fields constructor
class BundleMap(val fields: Seq[BundleFieldBase]) extends Record with CustomBulkAssignable {
  // All fields must have distinct key.names
  require(fields.map(_.key.name).distinct.size == fields.size)

  val elements: ListMap[String, Data] = ListMap(fields.map { bf => bf.key.name -> chisel3.experimental.DataMirror.internal.chiselTypeClone(bf.data) } :_*)
  override def cloneType: this.type = {
    try {
      this.getClass.getConstructors.head.newInstance(fields).asInstanceOf[this.type]
    } catch {
      case e: java.lang.IllegalArgumentException =>
        throw new Exception("Unable to use BundleMap.cloneType on " +
          this.getClass + ", probably because " + this.getClass +
          " does not have a constructor accepting BundleFields.  Consider overriding " +
          "cloneType() on " + this.getClass, e)
    }
  }

  // A BundleMap is best viewed as a map from BundleKey to Data
  def keydata: Seq[(BundleKeyBase, Data)] = (fields zip elements) map { case (field, (_, data)) => (field.key, data) }

  def apply[T <: Data](key: BundleKey[T]): T = elements(key.name).asInstanceOf[T]
  def lift [T <: Data](key: BundleKey[T]): Option[T] = elements.get(key.name).map(_.asInstanceOf[T])

  def apply(key: BundleKeyBase): Data         = elements(key.name)
  def lift (key: BundleKeyBase): Option[Data] = elements.get(key.name)

  // Create a new BundleMap with only the selected Keys retained
  def subset(fn: BundleKeyBase => Boolean): BundleMap = {
    val out = Wire(BundleMap(fields.filter(x => fn(x.key))))
    out :<= this
    out
  }

  // Assign all outputs of this from either:
  //   outputs of that (if they exist)
  //   or the default value for the BundleField
  def assignL(that: CustomBulkAssignable): Unit = { // this/bx :<= that/by
    require(that.isInstanceOf[BundleMap], s"Illegal attempt to drive BundleMap $this :<= non-BundleMap $that")
    val bx = this
    val by = that.asInstanceOf[BundleMap]
    val hy = mutable.HashMap(by.elements.toList:_*)
    (bx.fields zip bx.elements) foreach { case (field, (_, vx)) =>
      hy.get(field.key.name) match {
        case Some(vy) => FixChisel3.descendL(vx, vy)
        case None => DataMirror.specifiedDirectionOf(vx) match {
          case SpecifiedDirection.Output => field.setDataDefault(vx)
          case SpecifiedDirection.Input => ()
          case _ => require(requirement = false, s"Attempt to assign $bx :<= $by, where RHS is missing directional field $field")
        }
      }
    }
    // it's ok to have excess elements in 'hy'
  }

  // Assign all inputs of that from either:
  //   inputs of this (if they exist)
  //   or the default value for the BundleField
  def assignR(that: CustomBulkAssignable): Unit = { // this/bx :=> that/by
    require(that.isInstanceOf[BundleMap], s"Illegal attempt to drive BundleMap $this :=> non-BundleMap $that")
    def bx = this
    def by = that.asInstanceOf[BundleMap]
    val hx = mutable.HashMap(bx.elements.toList:_*)
    (by.fields zip by.elements) foreach { case (field, (_, vy)) =>
      hx.get(field.key.name) match {
        case Some(vx) => FixChisel3.descendR(vx, vy)
        case None => DataMirror.specifiedDirectionOf(vy) match {
          case SpecifiedDirection.Output => ()
          case SpecifiedDirection.Input => field.setDataDefault(vy)
          case _ => require (requirement = false, s"Attempt to assign $bx :=> $by, where LHS is missing directional field $field")
        }
      }
    }
    // it's ok to have excess elements in 'hx'
  }

  // Assign only those outputs of this which exist as outputs in that
  def partialAssignL(that: BundleMap): Unit = {
    val h = mutable.HashMap(that.keydata:_*)
    keydata foreach { case (key, vx) =>
      h.get(key).foreach { vy => FixChisel3.descendL(vx, vy) }
    }
  }

  // Assign only those inputs of that which exist as inputs in this
  def partialAssignR(that: BundleMap): Unit = {
    val h = mutable.HashMap(keydata:_*)
    that.keydata foreach { case (key, vy) =>
      h.get(key).foreach { vx => FixChisel3.descendL(vx, vy) }
    }
  }
}

object BundleMap {
  def apply(fields: Seq[BundleFieldBase] = Nil) = new BundleMap(fields)
}

trait CustomBulkAssignable {
  def assignL(that: CustomBulkAssignable): Unit // Custom implementation of :<=
  def assignR(that: CustomBulkAssignable): Unit // Custom implementaiton of :=>
}

// Implement the primitives of bulk assignment, :<= and :=>
object FixChisel3 {
  // Used by :<= for child elements to switch directionality
  def descendL(x: Data, y: Data): Unit = {
    DataMirror.specifiedDirectionOf(x) match {
      case SpecifiedDirection.Unspecified => assignL(x, y)
      case SpecifiedDirection.Output      => assignL(x, y); assignR(y, x)
      case SpecifiedDirection.Input       => ()
      case SpecifiedDirection.Flip        => assignR(y, x)
    }
  }

  // Used by :=> for child elements to switch directionality
  def descendR(x: Data, y: Data): Unit = {
    DataMirror.specifiedDirectionOf(y) match {
      case SpecifiedDirection.Unspecified => assignR(x, y)
      case SpecifiedDirection.Output      => ()
      case SpecifiedDirection.Input       => assignL(y, x); assignR(x, y)
      case SpecifiedDirection.Flip        => assignL(y, x)
    }
  }

  // The default implementation of 'x :<= y'
  // Assign all output fields of x from y
  def assignL(x: Data, y: Data): Unit = {
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
  def assignR(x: Data, y: Data): Unit = {
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

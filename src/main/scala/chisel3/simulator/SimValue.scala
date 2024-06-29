// SPDX-License-Identifier: Apache-2.0

package chisel3
package simulator

import chisel3.util.log2Ceil
import chisel3.experimental.SourceInfo

import svsim.Simulation

trait SimValue {}

object SimValue {
//  def toString[T <: Data](t: T): SimValue = {
//    require(t.isLit)
//    t match {
//      case v: Vec[_] =>
//        VecValue(v, v.map(apply(_)))
//      case b: Record =>
//        BundleValue(b, b.elements.map { case (name, data) => name -> apply(data) }.toMap)
//      case e: Element =>
//        LeafValue(e, e.widthOption.getOrElse(log2Ceil(e.litValue)), e.litValue)
//    }
//  }
}

case class LeafValue(bitCount: Int, asBigInt: BigInt, isSigned: Boolean = false)(implicit val sourceInfo: SourceInfo)
    extends SimValue {
  override def toString = asBigInt.toString
}
object LeafValue {
  def fromSimulationValue(sv: Simulation.Value, isSigned: Boolean = false) =
    LeafValue(sv.bitCount, sv.asBigInt, isSigned)
}

case class VecValue(value: Seq[SimValue])(implicit val sourceInfo: SourceInfo) extends SimValue {

  override def toString = s"[${value.mkString(", ")}]"
}

case class BundleValue(value: Map[String, SimValue])(implicit val sourceInfo: SourceInfo) extends SimValue {

  type K = String
  type V = SimValue

  def get(key: K): Option[V] = value.get(key)

  def getOrElse[V1 >: V](key: K, default: => V1): V1 = get(key) match {
    case Some(v) => v
    case None    => default
  }

  @throws[NoSuchElementException]
  def apply(key: K): V = get(key).get

  override def toString = s"{${value.map { case (k, v) => s"$k: $v" }.mkString(", ")}}"
}

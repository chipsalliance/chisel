package chisel3
package simulator

import chisel3.util.log2Ceil
import chisel3.experimental.SourceInfo

import svsim.Simulation

// trait HierarchicalValue {
//   val gen: Data
// }

// object HierarchicalValue {
//   def apply[T <: Data](t: T): HierarchicalValue = {
//     require(t.isLit)
//     t match {
//       case v: Vec[_] =>
//         VecValue(v, v.map(apply(_)))
//       case b: Record =>
//         BundleValue(b, b.elements.map { case (name, data) => name -> apply(data) }.toMap)
//       case e: Element =>
//         LeafValue(e, Simulation.Value(e.widthOption.getOrElse(log2Ceil(e.litValue)), e.litValue))
//     }
//   }
// }

// case class LeafValue[T <: Element](gen: T, value: Simulation.Value)(implicit sourceInfo: SourceInfo)
//     extends HierarchicalValue {
//   def toData: T = value.asBigInt.U(value.bitCount.W).asTypeOf(gen)
//   override def toString = s"${value.asBigInt}"
// }
// case class VecValue[T <: Vec[_]](gen: T, value: Seq[HierarchicalValue])(implicit sourceInfo: SourceInfo)
//     extends HierarchicalValue {

//   override def toString = s"[${value.mkString(", ")}]"
// }
// case class BundleValue[T <: Record](gen: T, value: Map[String, HierarchicalValue])(implicit sourceInfo: SourceInfo)
//     extends HierarchicalValue { // with MapOps[String, HierarchicalValue, HashMap, HashMap[String, HierarchicalValue]] {

//   type K = String
//   type V = HierarchicalValue

//   def get(key: K): Option[V] = value.get(key)

//   def getOrElse[V1 >: V](key: K, default: => V1): V1 = get(key) match {
//     case Some(v) => v
//     case None    => default
//   }

//   @throws[NoSuchElementException]
//   def apply(key: K): V = get(key).get

//   override def toString = s"{${value.map { case (k, v) => s"$k: $v" }.mkString(", ")}}"
// }

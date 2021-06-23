// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.dataview

import chisel3._
import chisel3.experimental.BaseModule
import chisel3.experimental.{BaseModule, FixedPoint}
import chisel3.internal.HasId
import chisel3.internal.firrtl._
import firrtl.annotations.ReferenceTarget

import scala.collection.mutable

import scala.annotation.implicitNotFound

/** Typeclass interface for getting elements of type [[Data]]
  *
  * This is needed for validating [[DataView]]s targeting type `A`.
  * Can be thought of as "can be the Target of a DataView".
  * @tparam A Type that has elements of type [[Data]]
  */
@implicitNotFound("Could not find implicit value for DataProduct[${A}].\nPlease see <docs link>")
trait DataProduct[-A] {
  /** Provides [[Data]] elements within some parent type
    *
    * @param a Parent type
    * @param path Hierarchical path to current signal (for error reporting)
    * @return Data elements and associated String paths (Strings for error reporting only!)
    */
  def dataIterator(a: A, path: String): Iterator[(Data, String)]
}

trait ModuleProduct[-A] extends DataProduct[A] {
  def contains(d: Data): Boolean
}

object DataProduct {
  implicit val dataDataProduct: DataProduct[Data] = new DataProduct[Data] {
    def dataIterator(a: Data, path: String): Iterator[(Data, String)] =
      getRecursiveFields(a, path).iterator
  }
  implicit val moduleDataProduct: DataProduct[BaseModule] = new DataProduct[BaseModule] {
    def dataIterator(a: BaseModule, path: String): Iterator[(Data, String)] = {
      getInternals(a).flatMap { case (name, data) => getRecursiveFields(data, path + name) }.iterator
    }
  }
  def getInternals(b: BaseModule): Seq[(String, Data)] = {
    val ports = b.getChiselPorts
    ports ++ b._component.get.asInstanceOf[DefModule].commands.flatMap {
      case d: DefPrim[_] => Seq(d.name -> d.id)
      case d: DefWire => Seq(d.name -> d.id)
      case d: DefReg => Seq(d.name -> d.id)
      case d: DefRegInit => Seq(d.name -> d.id)
      //case d: DefMemory(sourceInfo: SourceInfo, id: HasId, t: Data, size: BigInt) extends Definition
      //case d: DefSeqMemory(sourceInfo: SourceInfo, id: HasId, t: Data, size: BigInt, readUnderWrite: fir.ReadUnderWrite.Value) extends Definition
      //case d: DefMemPort[T <: Data](sourceInfo: SourceInfo, id: T, source: Node, dir: MemPortDirection, index: Arg, clock: Arg) extends Definition
      case d: DefInstance => getInternals(d.id)
      case _ => Nil
    }
  }
  implicit val instanceDataProduct: DataProduct[Instance[_ <: BaseModule]] = new DataProduct[Instance[_ <: BaseModule]] {
    def dataIterator(a: Instance[_ <: BaseModule], path: String): Iterator[(Data, String)] = {
      // Note that this is bad because we are basically cloning the entire template (which we are trying to avoid)
      // The plan is to fix this by a bounded id check
      val templateInternals = getInternals(a.template)
      val instanceInternals = templateInternals.map { case (name, data: Data) =>
        name -> a.apply[Data]{a: Any => data}
      }
      instanceInternals.flatMap { case (name, data) => getRecursiveFields(data, path + name) }.iterator
    }
  }

  def getData(b: BaseModule, path: String): Iterator[(Data, String)] = {
    (b._component.get match {
      case d: DefModule => d.commands.flatMap {
        case r: Definition => r.id match {
          case d: Data => getRecursiveFields(d, path)
          case _ => Nil
        }
        case _ => Nil
      } ++ d.ports.flatMap(p => getRecursiveFields(p.id, path))
    }).iterator
  }
}

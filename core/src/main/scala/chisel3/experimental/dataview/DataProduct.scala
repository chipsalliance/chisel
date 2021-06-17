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
      a.getChiselPorts.flatMap { case (name, data) => getRecursiveFields(data, path + name) }.toIterator
    }
  }
  implicit val instanceDataProduct: DataProduct[Instance[_]] = new DataProduct[Instance[_]] {
    def dataIterator(a: Instance[_], path: String): Iterator[(Data, String)] = {
      //a.getChiselPorts.flatMap { case (name, data) => getRecursiveFields(data, path + name) }.toIterator
      (a.io.elements.map(_._2).flatMap {
        x => getRecursiveFields(x, path)
      }).iterator
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

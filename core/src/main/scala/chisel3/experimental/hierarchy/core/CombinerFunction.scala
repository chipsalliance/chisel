// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core

import scala.collection.mutable.HashMap
import scala.collection.mutable
import scala.reflect.runtime.universe.TypeTag
import scala.language.experimental.macros
import chisel3.internal.sourceinfo.{DefinitionTransform, InstanceTransform}
import java.util.IdentityHashMap
import chisel3.experimental.hierarchy.func

sealed trait CombinerFunction { self =>
  val className = this.getClass.toString.split(" ")(1)
  override def toString = className
  try {
    Class.forName(className)
  } catch {
    case e: Exception => throw e
  }

  type I
  type O
  def apply(i:      List[I]): O
  def applyAny(any: List[Any]): Any = apply(any.asInstanceOf[List[I]])
}

trait CustomCombinerFunction[-I, +O] extends CombinerFunction {}

object CombinerFunction {
  def build(className: String, args: List[Any]): CombinerFunction = {
    Class.forName(className).getConstructor(classOf[List[Any]]).newInstance(args).asInstanceOf[CombinerFunction]
  }

}

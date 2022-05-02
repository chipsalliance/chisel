// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core

import scala.collection.mutable.HashMap
import scala.collection.mutable
import scala.reflect.runtime.universe.TypeTag
import scala.language.experimental.macros
import chisel3.internal.sourceinfo.{DefinitionTransform, InstanceTransform}
import java.util.IdentityHashMap
import chisel3.experimental.hierarchy.func

sealed trait ParameterFunction { self =>
  val className = this.getClass.toString.split(" ")(1)
  override def toString = className
  try {
    Class.forName(className)
  } catch {
    case e: Exception => throw e
  }

  type I
  type O
  def apply(i:      I): O
  def applyAny(any: Any): Any = apply(any.asInstanceOf[I])
}

case class Identity() extends ParameterFunction {
  type I = Any
  type O = Any
  def apply(i: Any): Any = i
}

trait CustomParameterFunction[-I, +O] extends ParameterFunction {
  def args: List[Any]
}

object ParameterFunction {
  def build(className: String, args: List[Any]): ParameterFunction = {
    Class.forName(className).getConstructor(classOf[List[Any]]).newInstance(args).asInstanceOf[ParameterFunction]
  }

  //@definitive def plus(n: Int)(i: Int): Int = n + i
}

//object ParameterFunction {
//  import scala.reflect.runtime.{universe => ru}
//  def run[I, O, T: TypeTag](i: I): O = {
//    val m = ru.runtimeMirror(getClass.getClassLoader)
//    println("HERE0")
//    val x = ru.typeOf[T].classSymbol
//    println(ru.typeOf[T].toString)
//    val o1 = m.reflect(x).instance
//    println(s"HERE1: $o1")
//    val o2 = o1.asInstanceOf[{ def applyAny(x: Any): Any }] // "structural typing"
//    println("HERE2")
//    o2.applyAny(i).asInstanceOf[O]
//  }
//}

import scala.reflect.runtime.{universe => ru}

//case class Func[T <: ParameterFunction[_, _] : TypeTag](t: T) extends ParameterFunction {
//  type I = t.I
//  type O = t.O
//  def apply(i: I): O = {
//  }
//}

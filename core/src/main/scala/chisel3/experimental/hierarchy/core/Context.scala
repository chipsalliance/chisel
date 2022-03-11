//// SPDX-License-Identifier: Apache-2.0
//
//package chisel3.experimental.hierarchy.core
//import java.util.IdentityHashMap
//
//// Wrapper Class
//final case class Context[+P, T](proxy: Proxy[P], contextual: Contextual[T], edit: T => T) {
//
//  private[chisel3] val cache = new IdentityHashMap[Any, Any]()
//
//  def _lookup[B, C](that: P => B)(
//    implicit contexter: Contexter[B],
//    macroGenerated:  chisel3.internal.MacroGenerated
//  ): contexter.R = {
//    val protoValue = that(this.proxy.proto)
//    val retValue = contexter(protoValue, this)
//    //cache.getOrElseUpdate(protoValue, retValue)
//    retValue
//  }
//}
//
//// Typeclass
//trait Contexter[V]  {
//  type R
//  def apply[P](value: V, context: Context[P]): R
//}
//
//// Default Typeclass Implementations
//object Contexter {
//}
//
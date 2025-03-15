// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

package object plugin {

  /** Used by Chisel's compiler plugin to automatically name signals
    * DO NOT USE in your normal Chisel code!!!
    *
    * @param name The name to use
    * @param nameMe The thing to be named
    * @tparam T The type of the thing to be named
    * @return The thing, but now named
    */
  @deprecated("Use chisel3.withName instead", "Chisel 7.0.0")
  def autoNameRecursively[T <: Any](name: String)(nameMe: => T): T = chisel3.withName[T](name)(nameMe)

  /** Used by Chisel's compiler plugin to automatically name signals
    * DO NOT USE in your normal Chisel code!!!
    *
    * @param names The names to use corresponding to interesting fields of the Product
    * @param nameMe The [[scala.Product]] to be named
    * @tparam T The type of the thing to be named
    * @return The thing, but with each member named
    */
  @deprecated("Use chisel3.withName instead", "Chisel 7.0.0")
  def autoNameRecursivelyProduct[T <: Product](names: List[Option[String]])(nameMe: => T): T =
    chisel3.withNames(names.map(_.getOrElse("")): _*)(nameMe)
}

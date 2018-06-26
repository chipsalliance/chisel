// See LICENSE for license details.

package chisel3.util

import scala.collection.mutable

class BindPrefixFactory(basePrefix: String) {
  val nameToPrefix: mutable.HashMap[String, String] = new mutable.HashMap
  var counter: Int = -1

  def newPrefix(name: String): String = {
    counter += 1
    s"Bind_${name}_${counter}_to_"
  }

  def apply(name: String): String = {
    nameToPrefix.get(name) match {
      case Some(prefix) => prefix
      case _ =>
        nameToPrefix(name) = newPrefix(name)
        nameToPrefix(name)
    }
  }
}

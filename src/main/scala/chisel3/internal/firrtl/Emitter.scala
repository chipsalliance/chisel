// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.firrtl

import scala.collection.immutable.LazyList // Needed for 2.12 alias
import firrtl.ir.Serializer

private[chisel3] object Emitter {
  def emit(circuit: Circuit): String = {
    val fcircuit = Converter.convertLazily(circuit)
    Serializer.serialize(fcircuit)
  }

  def emitLazily(circuit: Circuit): Iterable[String] = {
    val result = LazyList(s"circuit ${circuit.name} :\n")
    val modules = circuit.components.view.map(Converter.convert)
    val moduleStrings = modules.flatMap { m =>
      Array(Serializer.serialize(m, 1), "\n\n")
    }
    result ++ moduleStrings
  }
}

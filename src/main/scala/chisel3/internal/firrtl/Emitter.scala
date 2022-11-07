// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.firrtl

import scala.collection.immutable.LazyList // Needed for 2.12 alias
import firrtl.ir.Serializer

private[chisel3] object Emitter {
  def emit(circuit: Circuit): String = {
    val fcircuit = Converter.convertLazily(circuit)
    Serializer.serialize(fcircuit)
  }

  def emitLazily(circuit: Circuit): Iterable[String] = new Iterable[String] {
    def iterator = {
      val prelude = Iterator(s"circuit ${circuit.name} :\n")
      val modules = circuit.components.iterator.map(Converter.convert)
      val moduleStrings = modules.flatMap { m =>
        Serializer.lazily(m, 1) ++ Seq("\n\n")
      }
      prelude ++ moduleStrings
    }
  }
}

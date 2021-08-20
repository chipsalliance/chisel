// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.firrtl
import firrtl.{ir => fir}

private[chisel3] object Emitter {
  def emit(circuit: Circuit): String = {
    val fcircuit = Converter.convertLazily(circuit)
    fir.Serializer.serialize(fcircuit)
  }
}


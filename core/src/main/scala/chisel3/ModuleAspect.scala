// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.internal.{Builder, PseudoModule}

/** Used by Chisel Aspects to inject Chisel code into modules, after they have been elaborated.
  * This is an internal API - don't use!
  *
  * It adds itself as an aspect to the module, which allows proper checking of connection and binding legality.
  *
  * @param module Module for which this object is an aspect of
  */
abstract class ModuleAspect private[chisel3] (module: RawModule) extends RawModule with PseudoModule {

  Builder.addAspect(module, this)

  override def circuitName: String = module.toTarget.circuit

  override def desiredName: String = module.name

  override val _namespace = module._namespace
}

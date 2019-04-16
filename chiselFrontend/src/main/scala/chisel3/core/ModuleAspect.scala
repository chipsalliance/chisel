package chisel3.core

import chisel3.internal.Builder

abstract class ModuleAspect(module: RawModule)(implicit moduleCompileOptions: CompileOptions) extends RawModule {

  Builder.addAspect(module, this)

  override def circuitName: String = module.toTarget.circuit

  override def desiredName: String = module.name

  override val _namespace = module._namespace
}


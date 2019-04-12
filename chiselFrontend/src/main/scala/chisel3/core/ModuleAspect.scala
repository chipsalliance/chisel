package chisel3.core

abstract class ModuleAspect(module: RawModule)(implicit moduleCompileOptions: CompileOptions) extends RawModule {
  override def circuitName: String = module.toTarget.circuit

  override def desiredName: String = module.name

  override val _namespace = module._namespace
}


package chisel3
package experimental

trait HasImplementation[T <: Handoff] extends hierarchy.core.HasImplementationInternal { self: BaseModule =>
  val implementation: Implementation[T]
  def handoff: T
  final def inject = {
    if(ranInject) {} else {
      ranInject = true
      chisel3.aop.injecting.inject(this)({x => internal.Builder.clearPrefix(); implementation.build(handoff) })
    }
  }
  private[chisel3] var ranInject = false
  final def hasImplementation = ranInject

  if(internal.Builder.buildImplementation) {
    inject
  }
}

trait Implementation[T <: Handoff] { self: Singleton =>
  def build(handoff: T): Unit
}

trait Handoff
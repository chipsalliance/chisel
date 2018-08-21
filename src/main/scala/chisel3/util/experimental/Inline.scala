// See LICENSE for license details.

package chisel3.util.experimental

import chisel3._
import chisel3.internal.InstanceId
import chisel3.experimental.{BaseModule, ChiselAnnotation, RunFirrtlTransform}
import firrtl.Transform
import firrtl.passes.{InlineAnnotation, InlineInstances}
import firrtl.transforms.{NoDedupAnnotation, FlattenAnnotation, Flatten}
import firrtl.annotations.{CircuitName, ModuleName, ComponentName, Annotation}

/** Inlines an instance of a module
  *
  * @example {{{
  * trait Internals { this: Module =>
  *   val io = IO(new Bundle{ val a = Input(Bool()) })
  * }
  * class Sub extends Module with Internals
  * trait HasSub { this: Module with Internals =>
  *   val sub = Module(new Sub)
  *   sub.io.a := io.a
  * }
  * /* InlineInstance is mixed directly into Foo's definition. Every instance
  *  * of this will be inlined. */
  * class Foo extends Module with Internals with InlineInstance with HasSub
  * /* Bar will, by default, not be inlined */
  * class Bar extends Module with Internals with HasSub
  * /* The resulting instances will be:
  *  - Top
  *  - Top.x$sub
  *  - Top.y$sub
  *  - Top.z
  *  - Top.z.sub */
  * class Top extends Module with Internals {
  *   val x = Module(new Foo)                     // x will be inlined
  *   val y = Module(new Bar with InlineInstance) // y will also be inlined
  *   val z = Module(new Bar)                     // z will not be inlined
  *   Seq(x, y, z).map(_.io.a := io.a)
  * }
  * }}}
  */
trait InlineInstance { self: BaseModule =>
  Seq(new ChiselAnnotation with RunFirrtlTransform {
        def toFirrtl: Annotation = InlineAnnotation(self.toNamed)
        def transformClass: Class[_ <: Transform] = classOf[InlineInstances] },
      new ChiselAnnotation {
        def toFirrtl: Annotation = NoDedupAnnotation(self.toNamed) })
    .map(chisel3.experimental.annotate(_))
}

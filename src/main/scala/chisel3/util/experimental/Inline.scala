// SPDX-License-Identifier: Apache-2.0

package chisel3.util.experimental

import chisel3._
import chisel3.experimental.BaseModule
import firrtl.passes.InlineAnnotation
import firrtl.transforms.{FlattenAnnotation, NoDedupAnnotation}
import firrtl.annotations.Annotation

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
  *  - Top.x\$sub
  *  - Top.y\$sub
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
  chisel3.experimental.annotate(self)(Seq(InlineAnnotation(self.toNamed), NoDedupAnnotation(self.toNamed)))
}

/** Inlines all instances of a module. If this module dedups with any other
  * module, instances of that other module will also be inlined.
  */
trait InlineInstanceAllowDedup { self: BaseModule =>
  chisel3.experimental.annotate(self)(Seq(InlineAnnotation(self.toNamed)))
}

/** Flattens an instance of a module
  *
  * @example {{{
  * trait Internals { this: Module =>
  *   val io = IO(new Bundle{ val a = Input(Bool()) })
  * }
  * class Foo extends Module with Internals with FlattenInstance
  * class Bar extends Module with Internals {
  *   val baz = Module(new Baz)
  *   baz.io.a := io.a
  * }
  * class Baz extends Module with Internals
  * /* The resulting instances will be:
  *      - Top
  *      - Top.x
  *      - Top.y
  *      - Top.z
  *      - Top.z.baz */
  * class Top extends Module with Internals {
  *   val x = Module(new Foo)                      // x will be flattened
  *   val y = Module(new Bar with FlattenInstance) // y will also be flattened
  *   val z = Module(new Bar)                      // z will not be flattened
  *   Seq(x, y, z).map(_.io.a := io.a)
  * }
  * }}}
  */
trait FlattenInstance { self: BaseModule =>
  chisel3.experimental.annotate(self)(Seq(FlattenAnnotation(self.toNamed), NoDedupAnnotation(self.toNamed)))
}

/** Flattens all instances of a module. If this module dedups with any other
  * module, instances of that other module will also be flattened.
  */
trait FlattenInstanceAllowDedup { self: BaseModule =>
  chisel3.experimental.annotate(self)(Seq(FlattenAnnotation(self.toNamed)))
}

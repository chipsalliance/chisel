// See LICENSE for license details.

package chisel3.aop

import chisel3.RawModule
import chisel3.experimental.{BaseModule, ChiselAnnotation, annotate}
import firrtl.annotations.Annotation

package object injecting {
  object inject {
    def apply[T <: BaseModule](module: T)(f: T => Unit) = {
      val ia = InjectingAspect(
        {x: BaseModule => Seq(module)},
        f
      )

      annotate(new ChiselAnnotation { override def toFirrtl: Annotation = ia })

    }

    def withScope[P <: RawModule, C <: RawModule](parent: P)(selectChildren: P => Iterable[C])(onChild: C => Unit) = {

      val ia = InjectingAspect(
        {x: RawModule =>
          val parents = Select.collectDeep(x) { case x: BaseModule if x == parent => x.asInstanceOf[P] }
          parents.flatMap {
            p => selectChildren(p)
          }
        },
        onChild
      )

      annotate(new ChiselAnnotation { override def toFirrtl: Annotation = ia })

    }
  }

}

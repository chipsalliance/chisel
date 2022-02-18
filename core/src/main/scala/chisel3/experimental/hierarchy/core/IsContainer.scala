package chisel3.experimental.hierarchy.core

trait IsContainer

trait Containerable[B <: IsTypeclass[_]] extends IsTypeclass[B] {
  type C
  def apply[A](value: B, context: Hierarchy[A]): C
}

object Containerable {
  import scala.language.higherKinds // Required to avoid warning for lookupIterable type parameter
  implicit def iterable[X, B <: IsTypeclass[X], F[_] <: Iterable[_]](implicit tp: B) = new Containerable[F[X]] {
    type C = F[tp.C]
    def apply[A](value: F[X], context: Hierarchy[A]): C = {
      value.asInstanceOf[Iterable[X]].map { x: X => tp.apply[A](x, context) }.asInstanceOf[C]
    }
  }
}

//  implicit def lookupIterable[B, F[_] <: Iterable[_]](
//    implicit sourceInfo: SourceInfo,
//    compileOptions:      CompileOptions,
//    lookupable:          Lookupable[B]
//  ) = new Lookupable[F[B]] {
//    type C = F[lookupable.C]
//    def definitionLookup[A](that: A => F[B], definition: Definition[A]): C = {
//      val ret = that(definition.proto).asInstanceOf[Iterable[B]]
//      ret.map { x: B => lookupable.definitionLookup[A](_ => x, definition) }.asInstanceOf[C]
//    }
//  }
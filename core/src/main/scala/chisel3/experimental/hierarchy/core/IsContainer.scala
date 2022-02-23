package chisel3.experimental.hierarchy.core

trait IsContainer

//trait Containerable[F, B <: IsTypeclass[_]] extends IsTypeclass[B] {
//  type C
//  def apply[A](value: B, hierarchy: Hierarchy[A])(implicit h: Hierarchicable[A]): C
//}
//
//object Containerable {
//  import scala.language.higherKinds // Required to avoid warning for lookupIterable type parameter
//  implicit def iterable[X, B <: IsTypeclass[X], F[_] <: Iterable[_]](implicit tp: B) = new Containerable[B] {
//    type C = F[tp.C]
//    def apply[A](value: F[X], context: Hierarchy[A])(implicit h: Hierarchicable[A]): C = {
//      value.asInstanceOf[Iterable[X]].map { x: X => tp.apply[A](x, context) }.asInstanceOf[C]
//    }
//  }
//}

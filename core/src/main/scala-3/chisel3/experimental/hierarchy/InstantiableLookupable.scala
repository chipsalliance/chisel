// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy

import scala.annotation.nowarn
import scala.quoted.*
import chisel3.experimental.SourceInfo
import chisel3.experimental.hierarchy.core.{Clone, Definition, Hierarchy, Instance, Lookupable, Proto}

// Synthesize `toInstance` call for types with the @instantiable annotation
extension [T](self: T) {
  transparent inline def toInstance: Instance[T] =
    ${ ToInstanceMacro.impl[T]('self) }
}

private[hierarchy] object ToInstanceMacro {
  // Bridge for the inline macro: the macro splices a call to this
  // non-inline helper so it can use the package-private Instance
  // constructor without violating the inline restriction
  def makeInstance[T](self: T): Instance[T] = new Instance(Proto(self))

  def impl[T: Type](self: Expr[T])(using q: Quotes): Expr[Instance[T]] = {
    import q.reflect.*
    val tpe = TypeRepr.of[T]
    val sym = tpe.typeSymbol
    val instantiableSym = TypeRepr.of[instantiable].typeSymbol
    val hasInstantiable =
      sym.annotations.exists(_.tpe.typeSymbol == instantiableSym) ||
        sym.typeRef.baseClasses.exists { bc =>
          bc.annotations.exists(_.tpe.typeSymbol == instantiableSym)
        }
    if (!hasInstantiable) {
      report.errorAndAbort(
        s"`.toInstance` is only available on classes/traits annotated with @instantiable, but ${tpe.show} is not"
      )
    }
    '{ ToInstanceMacro.makeInstance[T]($self) }
  }
}

// HierarchyLookup calls `synthesize` when the type of a field is
// annotated as @instantiable but no Lookupable instance can be summoned
private[hierarchy] object InstantiableLookupable {
  @nowarn("cat=deprecation")
  def synthesize[B](using SourceInfo): Lookupable.Aux[B, Instance[B]] =
    new Lookupable[B] {
      type C = Instance[B]
      override def definitionLookup[A](that: A => B, definition: Definition[A]): C =
        impl(that, definition)
      override def instanceLookup[A](that: A => B, instance: Instance[A]): C =
        impl(that, instance)
      private def impl[A](that: A => B, hierarchy: Hierarchy[A]): C = {
        val ret = that(hierarchy.proto)
        val underlying = new InstantiableClone[B] {
          val getProto = ret
          lazy val _innerContext: Hierarchy[?] = hierarchy
        }
        new Instance(Clone(underlying))
      }
    }
}

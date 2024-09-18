// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy

import scala.reflect.runtime.{universe => ru}
import scala.collection.mutable

import chisel3._
import chisel3.experimental.{BaseModule, SourceInfo, UnlocatableSourceInfo}
import chisel3.reflect.DataMirror
import chisel3.reflect.DataMirror.internal.isSynthesizable
import chisel3.internal.Builder

private[chisel3] trait InstantiateImpl {

  // Data uses referential equality by default, but for looking up Data in the cache, we need to use
  // structural equality for Data unbound types and literal values
  // Note that this is somewhat incomplete because we can't handle the general case of user-defined
  // types that contain Data.
  // A more complete and robust solution would require either refining hashCode and equality on Data
  // to have the desired behavior, or using a different Map implementation that uses typeclasses for
  // equality and hashcode (eg. cats-collections-core's HashMap)
  private class DataBox(private val d: Data) {

    private def convertDataForHashing(data: Data): Any = data match {
      // Using toString is a bit weird but it works
      case elt: Element => elt.toString
      case rec: Record  =>
        // Purely structural, actual class of Record isn't involved
        rec.elements.map { case (name, data) => name -> convertDataForHashing(data) }
      case vec: Vec[_] =>
        // We could map on elements but that's a lot of duplicated work for a type
        // Note that empty vecs of the same type will give same hash value, probably should be equal
        // as well, but Vec.typeEquivalent checks sample_element so they will not be equal
        ("Vec", vec.size, vec.headOption.map(convertDataForHashing(_)))
    }

    override def hashCode: Int = convertDataForHashing(d).hashCode

    // If literals, check same types and equal
    // If types, chck same types
    // If bound, fall back to normal equality
    override def equals(that: Any): Boolean = {
      that match {
        case that: DataBox =>
          // def because it's not needed by non-literal but is synthesizable check
          def sameTypes = DataMirror.checkTypeEquivalence(this.d, that.d)
          // Lits are synthesizable so must check them first
          if (this.d.isLit) {
            that.d.isLit &&
            (this.d.litValue == that.d.litValue) &&
            sameTypes
          } else if (isSynthesizable(this.d)) {
            this.d.equals(that.d)
          } else {
            // We have checked that this.d is not synthesizable but need to check that.d as well
            sameTypes && !isSynthesizable(that.d)
          }

        case _ => false
      }
    }
  }

  // Recursively box all Data (by traversing Products and Iterables) in DataBoxes
  private def boxAllData(a: Any): Any = a match {
    case d: Data => new DataBox(d) // Must check this before Iterable because Vec is Iterable
    // Must check before Product, because many Iterables are Products, but can still be equal, eg. List(1) == Vector(1)
    case it: Iterable[_] => it.map(boxAllData(_))
    case p:  Product     => Vector(p.getClass) ++ p.productIterator.map(boxAllData(_))
    case other => other
  }

  import chisel3.internal.BuilderContextCache
  // Include type of module in key since different modules could have the same arguments
  private case class CacheKey[A <: BaseModule](args: Any, tt: ru.WeakTypeTag[A])
      extends BuilderContextCache.Key[Definition[A]]

  protected def _instanceImpl[K, A <: BaseModule: ru.WeakTypeTag](
    args: K,
    f:    K => A
  )(
    implicit sourceInfo: SourceInfo
  ): Instance[A] = Instance.do_apply(_definitionImpl(args, f))(sourceInfo)

  /** This is not part of the public API, do not call directly! */
  protected def _definitionImpl[K, A <: BaseModule: ru.WeakTypeTag](
    args: K,
    f:    K => A
  ): Definition[A] = {
    Builder.contextCache
      .getOrElseUpdate(
        CacheKey(boxAllData(args), implicitly[ru.WeakTypeTag[A]]), {
          // The definition needs to have no source locator because otherwise it will be unstably
          // derived from the first invocation of Instantiate for the particular Module
          Definition.do_apply(f(args))(UnlocatableSourceInfo)
        }
      )
      .asInstanceOf[Definition[A]]
  }
}

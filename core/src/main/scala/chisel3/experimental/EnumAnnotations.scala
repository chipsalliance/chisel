// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

import scala.annotation.nowarn
import chisel3._
import firrtl.annotations._

@nowarn("msg=Avoid custom annotations")
object EnumAnnotations {

  /** An annotation for strong enum instances that are ''not'' inside of Vecs
    *
    * @param target the enum instance being annotated
    * @param enumTypeName the name of the enum's type (e.g. ''"mypackage.MyEnum"'')
    */
  @deprecated("Enum annotations will be removed in Chisel 7.0.", "Chisel 6.7.0")
  case class EnumComponentAnnotation(target: Named, enumTypeName: String) extends SingleTargetAnnotation[Named] {
    def duplicate(n: Named): EnumComponentAnnotation = this.copy(target = n)
  }

  @deprecated("Enum annotations will be removed in Chisel 7.0.", "Chisel 6.7.0")
  case class EnumComponentChiselAnnotation(target: InstanceId, enumTypeName: String) extends ChiselAnnotation {
    def toFirrtl: EnumComponentAnnotation = EnumComponentAnnotation(target.toNamed, enumTypeName)
  }

  /** An annotation for Vecs of strong enums.
    *
    * The ''fields'' parameter deserves special attention, since it may be difficult to understand. Suppose you create a the following Vec:
    *
    *               {{{
    *               VecInit(new Bundle {
    *                 val e = MyEnum()
    *                 val b = new Bundle {
    *                   val inner_e = MyEnum()
    *                 }
    *                 val v = Vec(3, MyEnum())
    *               }
    *               }}}
    *
    *               Then, the ''fields'' parameter will be: ''Seq(Seq("e"), Seq("b", "inner_e"), Seq("v"))''. Note that for any Vec that doesn't contain Bundles, this field will simply be an empty Seq.
    *
    * @param target the Vec being annotated
    * @param typeName the name of the enum's type (e.g. ''"mypackage.MyEnum"'')
    * @param fields a list of all chains of elements leading from the Vec instance to its inner enum fields.
    */
  @deprecated("Enum annotations will be removed in Chisel 7.0.", "Chisel 6.7.0")
  case class EnumVecAnnotation(target: Named, typeName: String, fields: Seq[Seq[String]])
      extends SingleTargetAnnotation[Named] {
    def duplicate(n: Named): EnumVecAnnotation = this.copy(target = n)
  }

  @deprecated("Enum annotations will be removed in Chisel 7.0.", "Chisel 6.7.0")
  case class EnumVecChiselAnnotation(target: InstanceId, typeName: String, fields: Seq[Seq[String]])
      extends ChiselAnnotation {
    override def toFirrtl: EnumVecAnnotation = EnumVecAnnotation(target.toNamed, typeName, fields)
  }

  /** An annotation for enum types (rather than enum ''instances'').
    *
    * @param typeName the name of the enum's type (e.g. ''"mypackage.MyEnum"'')
    * @param definition a map describing which integer values correspond to which enum names
    */
  @deprecated("Enum annotations will be removed in Chisel 7.0.", "Chisel 6.7.0")
  case class EnumDefAnnotation(typeName: String, definition: Map[String, BigInt]) extends NoTargetAnnotation

  @deprecated("Enum annotations will be removed in Chisel 7.0.", "Chisel 6.7.0")
  case class EnumDefChiselAnnotation(typeName: String, definition: Map[String, BigInt]) extends ChiselAnnotation {
    override def toFirrtl: Annotation = EnumDefAnnotation(typeName, definition)
  }
}

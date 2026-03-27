// SPDX-License-Identifier: Apache-2.0

package chisel3.choice

import scala.language.experimental.macros
import scala.reflect.macros.blackbox.Context

private[chisel3] trait DynamicGroupFactoryIntf {
  implicit def materializeDynamicGroupFactory[T <: DynamicGroup]: DynamicGroup.Factory[T] =
    macro DynamicGroupMacros.materializeFactory[T]
}

private[chisel3] object DynamicGroupMacros {
  def materializeFactory[T <: DynamicGroup: c.WeakTypeTag](c: Context): c.Tree = {
    import c.universe._

    val dynamicGroupTpe = weakTypeOf[DynamicGroup]
    val caseTpe = weakTypeOf[Case]
    val sourceInfoTpe = weakTypeOf[chisel3.experimental.SourceInfo]
    val targetTpe = weakTypeOf[T]
    val targetSym = targetTpe.typeSymbol

    if (!targetSym.isClass || !targetSym.asClass.isTrait) {
      c.abort(
        c.enclosingPosition,
        s"DynamicGroup can only be materialized for traits, got: ${targetTpe.typeSymbol.fullName}"
      )
    }
    if (!(targetTpe <:< dynamicGroupTpe)) {
      c.abort(c.enclosingPosition, s"${targetTpe.typeSymbol.fullName} must extend chisel3.choice.DynamicGroup")
    }

    val caseNames = targetTpe.decls.toList.collect {
      case module: ModuleSymbol if module.typeSignature <:< caseTpe =>
        module.name.decodedName.toString.trim
    }.reverse

    if (caseNames.isEmpty) {
      c.abort(c.enclosingPosition, s"${targetSym.fullName} must declare at least one `object ... extends Case`")
    }

    val caseNameTrees = caseNames.map(name => Literal(Constant(name)))

    q"""
      new _root_.chisel3.choice.DynamicGroup.Factory[$targetTpe] {
        override val caseNames: _root_.scala.Seq[String] = _root_.scala.Seq(..$caseNameTrees)
        override def create()(implicit sourceInfo: $sourceInfoTpe): $targetTpe =
          new $targetTpe {}
      }
    """
  }
}

// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.plugin

import dotty.tools.dotc.*
import dotty.tools.dotc.ast.tpd
import dotty.tools.dotc.ast.tpd.*
import dotty.tools.dotc.core.Contexts.*
import dotty.tools.dotc.core.Symbols.*
import dotty.tools.dotc.core.Names.TermName
import dotty.tools.dotc.core.StdNames.*
import dotty.tools.dotc.core.Constants.Constant
import dotty.tools.dotc.typer.TyperPhase
import dotty.tools.dotc.plugins.{PluginPhase, StandardPlugin}
import dotty.tools.dotc.transform.{Pickler, PostTyper, Erasure}
import dotty.tools.dotc.core.Types.*

import scala.annotation.tailrec

class ChiselComponent extends StandardPlugin {
  val name: String = "ChiselComponent"
  override val description: String = "Chisel's type-specific naming"

  override def init(options: List[String]): List[PluginPhase] = {
    (new ChiselComponentPhase) :: Nil
  }
}

@tailrec
def printTreeString(t: String, accum: String = "", indent: Int = 0): String = {
  val parenop: Option[Char] = t.find { c => c == '(' || c == ')'}
  parenop match {
    case Some('(') => {
      val splitstr = t.span { c => c != parenop.get }
      printTreeString(
        splitstr._2 stripPrefix("("),
        accum ++ (" " * indent) ++ (splitstr._1 stripPrefix(",")) ++ "\n",
        indent + 2
      )
    }
    case Some(')') => {
      val splitstr = t.span { c => c != parenop.get }
      def newstr: String = splitstr._1 match {
        case "" => accum
        case _ => accum ++ (" " * indent) ++ (splitstr._1 stripPrefix(",")) ++ "\n"
      }
      printTreeString(
        splitstr._2 stripPrefix(")"),
        newstr,
        indent - 2
      )
    }
    case _ => accum
  }
}

class ChiselComponentPhase extends PluginPhase {

  val phaseName: String = "chiselComponentPhase"
  override val runsAfter = Set(TyperPhase.name)


  override def transformValDef(tree: tpd.ValDef)(using Context): tpd.Tree = {
    // println(s"tpe: ${isNamedComponent(tree.tpt.tpe)}")
    val dataTpe = requiredClassRef("chisel3.Data")
    val memBaseTpe = requiredClassRef("chisel3.MemBase")
    val verifTpe = requiredClassRef("chisel3.VerificationStatement")
    val dynObjTpe = requiredClassRef("chisel3.Disable")
    val affectsTpe = requiredClassRef("chisel3.experimental.AffectsChiselName")

    val moduleTpe = requiredClassRef("chisel3.experimental.BaseModule")
    val instTpe = requiredClassRef("chisel3.experimental.hierarchy.Instance")
    val prefixTpe = requiredClassRef("chisel3.experimental.AffectsChiselPrefix")

    val bundleTpe = requiredClassRef("chisel3.Bundle")

    def isNamedComponent(t: Type): Boolean = {
      t <:< dataTpe ||
      t <:< memBaseTpe ||
      t <:< verifTpe ||
      t <:< dynObjTpe ||
      t <:< affectsTpe
    }

    val valName: String = tree.name.show
    val nameLiteral = Literal(Constant(valName))
    val pluginModule = requiredModule("chisel3.internal.plugin")
    val autoNameMethod = pluginModule.requiredMethod("autoNameRecursively")
    val prefixModule = requiredModule("chisel3.experimental.prefix")

    val compTpe = tree.tpt.tpe

    if ((compTpe <:< dataTpe || compTpe <:< prefixTpe) && !(compTpe <:< bundleTpe)) {
      val newRhs = tpd.ref(pluginModule).select(autoNameMethod).appliedToType(tree.rhs.tpe).appliedTo(nameLiteral).appliedTo(tree.rhs)
      val prefixLiteral = if (valName.head == '_') Literal(Constant(valName.tail)) else Literal(Constant(valName))
      val prefixApplyMethod = prefixModule.requiredMethod("applyString")
      val prefixed = tpd.ref(prefixModule).select(prefixApplyMethod).appliedToType(tree.rhs.tpe).appliedTo(prefixLiteral).appliedTo(newRhs)

      if (isNamedComponent(compTpe)) {
        tpd.cpy.ValDef(tree)(rhs = newRhs)
      } else {
        tpd.cpy.ValDef(tree)(rhs = prefixed)
      }
    }
    else {
      tree
    }
  }
}

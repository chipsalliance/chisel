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

    val dataTpe = requiredClassRef("chisel3.Data")
    val memBaseTpe = requiredClassRef("chisel3.MemBase")
    val verifTpe = requiredClassRef("chisel3.VerificationStatement")
    val dynObjTpe = requiredClassRef("chisel3.Disable")
    val affectsTpe = requiredClassRef("chisel3.experimental.AffectsChiselName")

    def isNamed(t: Type): Boolean = {
      t <:< dataTpe ||
      t <:< memBaseTpe ||
      t <:< verifTpe ||
      t <:< dynObjTpe ||
      t <:< affectsTpe
    }
    val valName: String = tree.name.show
    println(s"tpe: ${isNamed(tree.tpt.tpe)}")

    val nameLiteral = Literal(Constant(valName))
    val pluginModule = requiredModule("chisel3.internal.plugin")
    val autoNameMethod = pluginModule.requiredMethod("autoNameRecursively")
    val newRhs = tpd.ref(pluginModule).select(autoNameMethod).appliedToType(tree.rhs.tpe).appliedTo(nameLiteral).appliedTo(tree.rhs)
    tpd.cpy.ValDef(tree)(rhs = newRhs)
    // case dd @ tpd.ValDef(name, tpt, rhs) =>
    //   println(s"found valdef: $name")
    //   val strName = name.toString.trim
    //   val prefix = if (strName.head == '_') strName.tail else strName
      // val prefixed = '{chisel3.experimental.prefix.apply[tpt.type](name=prefix)(f=tpt)}
      // val k = chisel3.internal.plugin.autoNameRecursively(strName)(prefixed)
      // println(s"K IS: $k")
    //   vTree
    // case _ => vTree
  }

  // override def transform(tree: Tree)(using ctx: Context): Tree = {
  //   println("in transform")
  //   tree
  // }
  // override def transformApply(tree: tpd.Tree)(implicit ctx: Context): tpd.Tree = tree match {
  //   case _ => {println(tree); tpd.EmptyTree}
  // }
}

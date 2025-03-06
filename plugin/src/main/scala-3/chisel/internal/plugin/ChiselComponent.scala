// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

import dotty.tools.dotc.ast.tpd
import dotty.tools.dotc.core.Contexts.Context
import dotty.tools.dotc.core.Phases.Phase
import dotty.tools.dotc.core.Decorators.*
import dotty.tools.dotc.core.StdNames.*
import dotty.tools.dotc.core.Symbols.*
import dotty.tools.dotc.CompilationUnit
import dotty.tools.dotc.plugins.* // StandardPlugin, PluginPhase
import dotty.tools.dotc.typer.TyperPhase
import dotty.tools.dotc.core.Names.{termName, typeName}
import dotty.tools.dotc.util.Spans.*
import dotty.tools.dotc.transform.{PickleQuotes, Staging, Pickler, Inlining}

import scala.annotation.tailrec
import scala.quoted.*

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
  // override val runsBefore = Set(Inlining.name)
  override def runOn(units: List[CompilationUnit])(using ctx: Context): List[CompilationUnit] = {
    println("naming modules")

    units.foreach { unit =>
      println(s"UNIT $unit")
      val tpdTree = unit.tpdTree
      // val transformedTree = transformTree(untpdTree)
      // println(s"TRANSFORMED TREE: $transformedTree")
      println("\n\n")
      // println(s"Type Tree: ${printTreeString(tpdTree.toString)}")
      // println(s"Type Tree: $tpdTree")
      // unit.tpdTree = tpdTree
    }
    units
  }

  // inline def prefixComp(prefix)
  override def transformValDef(vTree: tpd.ValDef)(using Context): tpd.Tree = vTree match {
    case dd @ tpd.ValDef(name, tpt, rhs) =>
      println(s"found valdef: $name")
      val strName = name.toString.trim
      val prefix = if (strName.head == '_') strName.tail else strName
      val prefixed = chisel3.experimental.prefix.apply[tpt.type](name=prefix)(f=tpt)
      val k = chisel3.internal.plugin.autoNameRecursively(strName)(prefixed)
      println(s"K IS: $k")
      vTree
    case _ => vTree
  }


  // override def transform(tree: Tree)(using ctx: Context): Tree = {
  //   println("in transform")
  //   tree
  // }
  // override def transformApply(tree: tpd.Tree)(implicit ctx: Context): tpd.Tree = tree match {
  //   case _ => {println(tree); tpd.EmptyTree}
  // }
}

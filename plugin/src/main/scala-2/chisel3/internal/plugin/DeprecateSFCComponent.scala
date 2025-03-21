// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.plugin

import scala.collection.mutable
import scala.reflect.internal.Flags
import scala.annotation.tailrec
import scala.tools.nsc
import scala.tools.nsc.{Global, Phase}
import scala.tools.nsc.plugins.PluginComponent
import scala.tools.nsc.transform.TypingTransformers

// Creates a warning for any code with an `import firrtl`, as this will be removed in Chisel 5.0 release.
class DeprecateSFCComponent(val global: Global, arguments: ChiselPluginArguments)
    extends PluginComponent
    with TypingTransformers {
  import global._
  val runsAfter:              List[String] = List[String]("typer")
  val phaseName:              String = "deprecatesfccomponent"
  def newPhase(_prev: Phase): DeprecateSFCComponent = new DeprecateSFCComponent(_prev)
  class DeprecateSFCComponent(prev: Phase) extends StdPhase(prev) {
    override def name: String = phaseName
    def apply(unit: CompilationUnit): Unit = {
      if (ChiselPlugin.runComponent(global, arguments)(unit)) {
        unit.body = new MyTypingTransformer(unit).transform(unit.body)
      }
    }
  }

  class MyTypingTransformer(unit: CompilationUnit) extends TypingTransformer(unit) {
    // The following trickery is necessary for using the new warning mechanism added in 2.13 from a compiler plugin
    // See https://github.com/scala/bug/issues/12258
    object WarningCategoryCompat {
      object Reporting {
        object WarningCategory {
          val Deprecation: Any = null
        }
      }
    }

    // Of type Reporting.WarningCategory.type, but we cannot explicit write it
    val WarningCategory = {
      import WarningCategoryCompat._

      {
        import scala.tools.nsc._
        Reporting.WarningCategory
      }
    }
    implicit final class GlobalCompat(self: DeprecateSFCComponent.this.global.type) {

      // Added in Scala 2.13.2 for configurable warnings
      object runReporting {
        def warning(pos: Position, msg: String, cat: Any, site: Symbol): Unit =
          reporter.warning(pos, msg)
      }
    }

    @tailrec private def isRootFirrtl(tree: Tree): Boolean = tree match {
      case Ident(name) if name.toString == "firrtl" => true
      case Select(expr, _)                          => isRootFirrtl(expr)
      case _                                        => false
    }

    @tailrec private def firrtlSymbol(sym: Symbol, rec: Int = 0): Boolean = sym match {
      case null | NoSymbol => false
      case sym if sym.name.toString == "firrtl" => true
      case _ if rec > 5 => throw new Exception(s"Stuck on ${showRaw(sym)}: ${sym.getClass()} with ${sym.enclosingPackage}") with scala.util.control.NoStackTrace
      case _ => firrtlSymbol(sym.enclosingPackage, rec + 1)
    }

    val debug = unit.source.file.name == "chisel-example.scala"

    private def warnOnSymbol(sym: Symbol): Unit = {
      // Can supress with adding "-Wconf:msg=firrtl:s" to scalacOptions
      global.runReporting.warning(
        sym.pos,
        s"Importing from firrtl is deprecated as of Chisel's 3.6.0 release.",
        WarningCategory.Deprecation,
        sym
      )
    }
  
    // Method called by the compiler to modify source tree
    override def transform(tree: Tree): Tree = tree match {
      case imp @ Import(expr: Tree, selectors: List[ImportSelector]) if isRootFirrtl(expr) =>
        warnOnSymbol(imp.symbol)
        super.transform(imp)
      // case tree if firrtlSymbol(tree.symbol) =>
      //   println(s"Warning on $tree : ${showRaw(tree)}")
      //   warnOnSymbol(tree.symbol)
      //   super.transform(tree)
      case sel @ Select(expr, _) if isRootFirrtl(expr) =>
        global.runReporting.warning(
          sel.pos,
          s"Importing from firrtl is deprecated as of Chisel's 3.6.0 release.",
          WarningCategory.Deprecation,
          sel.symbol
        )
        super.transform(sel)
      case _ => 
        // println(tree)
        // println(showRaw(tree))
        super.transform(tree)
    }
  }
}

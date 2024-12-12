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
  val runsAfter: List[String] = List[String]("typer")
  val phaseName: String = "deprecatesfccomponent"
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
    implicit final class GlobalCompat(
      self: DeprecateSFCComponent.this.global.type) {

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

    // Method called by the compiler to modify source tree
    override def transform(tree: Tree): Tree = tree match {
      case imp @ Import(expr: Tree, selectors: List[ImportSelector]) if isRootFirrtl(expr) =>
        // Can supress with adding "-Wconf:msg=firrtl:s" to scalacOptions
        global.runReporting.warning(
          imp.symbol.pos,
          s"Importing from firrtl is deprecated as of Chisel's 3.6.0 release.",
          WarningCategory.Deprecation,
          imp.symbol
        )
        super.transform(imp)
      case _ => super.transform(tree)
    }
  }
}

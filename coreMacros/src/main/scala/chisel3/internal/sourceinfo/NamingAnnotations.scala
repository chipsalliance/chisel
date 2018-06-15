// See LICENSE for license details.

// Transform implementations for name-propagation related annotations.
//
// Helpful references:
// http://docs.scala-lang.org/overviews/quasiquotes/syntax-summary.html#definitions
//   for quasiquote structures of various Scala structures
// http://jsuereth.com/scala/2009/02/05/leveraging-annotations-in-scala.html
//   use of Transformer
// http://www.scala-lang.org/old/sites/default/files/sids/rytz/Wed,%202010-01-27,%2015:10/annots.pdf
//   general annotations reference

package chisel3.internal.naming

import scala.reflect.macros.whitebox.Context
import scala.language.experimental.macros
import scala.annotation.StaticAnnotation
import scala.annotation.compileTimeOnly

// Workaround for https://github.com/sbt/sbt/issues/3966
object DebugTransforms
class DebugTransforms(val c: Context) {
  import c.universe._

  /** Passthrough transform that prints the annottee for debugging purposes.
    * No guarantees are made on what this annotation does, and it may very well change over time.
    *
    * The print is warning level to make it visually easier to spot, as well as a reminder that
    * this annotation should not make it to production / committed code.
    */
  def dump(annottees: c.Tree*): c.Tree = {
    val combined = annottees.map({ tree => show(tree) }).mkString("\r\n\r\n")
    annottees.foreach(tree => c.warning(c.enclosingPosition, s"Debug dump:\n$combined"))
    q"..$annottees"
  }

  /** Passthrough transform that prints the annottee as a tree for debugging purposes.
    * No guarantees are made on what this annotation does, and it may very well change over time.
    *
    * The print is warning level to make it visually easier to spot, as well as a reminder that
    * this annotation should not make it to production / committed code.
    */
  def treedump(annottees: c.Tree*): c.Tree = {
    val combined = annottees.map({ tree => showRaw(tree) }).mkString("\r\n")
    annottees.foreach(tree => c.warning(c.enclosingPosition, s"Debug tree dump:\n$combined"))
    q"..$annottees"
  }
}

// Workaround for https://github.com/sbt/sbt/issues/3966
object NamingTransforms
class NamingTransforms(val c: Context) {
  import c.universe._
  import Flag._

  val globalNamingStack = q"_root_.chisel3.internal.DynamicNamingStack()"

  /** Base transformer that provides the val name transform.
    * Should not be instantiated, since by default this will recurse everywhere and break the
    * naming context variable bounds.
    */
  trait ValNameTransformer extends Transformer {
    val contextVar: TermName

    override def transform(tree: Tree) = tree match {
      // Intentionally not prefixed with $mods, since modifiers usually mean the val definition
      // is in a non-transformable location, like as a parameter list.
      // TODO: is this exhaustive / correct in all cases?
      case q"val $tname: $tpt = $expr" => {
        val TermName(tnameStr: String) = tname
        val transformedExpr = super.transform(expr)
        q"val $tname: $tpt = $contextVar.name($transformedExpr, $tnameStr)"
      }
      case other => super.transform(other)
    }
  }

  /** Module-specific val name transform, containing logic to prevent from recursing into inner
    * classes and applies the naming transform on inner functions.
    */
  class ModuleTransformer(val contextVar: TermName) extends ValNameTransformer {
    override def transform(tree: Tree) = tree match {
      case q"$mods class $tpname[..$tparams] $ctorMods(...$paramss) extends { ..$earlydefns } with ..$parents { $self => ..$stats }" =>
        tree  // don't recurse into inner classes
      case q"$mods trait $tpname[..$tparams] extends { ..$earlydefns } with ..$parents { $self => ..$stats }" =>
        tree  // don't recurse into inner classes
      case q"$mods def $tname[..$tparams](...$paramss): $tpt = $expr" => {
        val Modifiers(_, _, annotations) = mods
        // don't apply naming transform twice
        val containsChiselName = annotations.map({q"new chiselName()" equalsStructure _}).fold(false)({_||_})
        // transforming overloaded initializers causes errors, and the transform isn't helpful
        val isInitializer = tname == TermName("<init>")
        if (containsChiselName || isInitializer) {
          tree
        } else {
          // apply chiselName transform by default
          val transformedExpr = transformHierarchicalMethod(expr)
          q"$mods def $tname[..$tparams](...$paramss): $tpt = $transformedExpr"
        }
      }
      case other => super.transform(other)
    }
  }

  /** Method-specific val name transform, handling the return case.
    */
  class MethodTransformer(val contextVar: TermName) extends ValNameTransformer {
    override def transform(tree: Tree) = tree match {
      // TODO: better error messages when returning nothing
      case q"return $expr" => q"return $globalNamingStack.pop_return_context($expr, $contextVar)"
      // Do not recurse into methods
      case q"$mods def $tname[..$tparams](...$paramss): $tpt = $expr" => tree
      case other => super.transform(other)
    }
  }

  /** Applies the val name transform to a module body. Pretty straightforward, since Module is
    * the naming top level.
    */
  def transformModuleBody(stats: List[c.Tree]) = {
    val contextVar = TermName(c.freshName("namingContext"))
    val transformedBody = (new ModuleTransformer(contextVar)).transformTrees(stats)

    q"""
    val $contextVar = $globalNamingStack.push_context()
    ..$transformedBody
    $contextVar.name_prefix("")
    $globalNamingStack.pop_context($contextVar)
    """
  }

  /** Applies the val name transform to a method body, doing additional bookkeeping with the
    * context to allow names to propagate and prefix through the function call stack.
    */
  def transformHierarchicalMethod(expr: c.Tree) = {
    val contextVar = TermName(c.freshName("namingContext"))
    val transformedBody = (new MethodTransformer(contextVar)).transform(expr)

    q"""{
      val $contextVar = $globalNamingStack.push_context()
      $globalNamingStack.pop_return_context($transformedBody, $contextVar)
    }
    """
  }

  /** Applies naming transforms to vals in the annotated module or method.
    *
    * For methods, a hierarchical naming transform is used, where it will try to give objects names
    * based on the call stack, assuming all functions on the stack are annotated as such and return
    * a non-AnyVal object. Does not recurse into inner functions.
    *
    * For modules, this serves as the root of the call stack hierarchy for naming purposes. Methods
    * will have chiselName annotations (non-recursively), but this does NOT affect inner classes.
    *
    * Basically rewrites all instances of:
    * val name = expr
    * to:
    * val name = context.name(expr, name)
    */
  def chiselName(annottees: c.Tree*): c.Tree = {
    var namedElts: Int = 0

    val transformed = annottees.map(annottee => annottee match {
      case q"$mods class $tpname[..$tparams] $ctorMods(...$paramss) extends { ..$earlydefns } with ..$parents { $self => ..$stats }" => {
        val transformedStats = transformModuleBody(stats)
        namedElts += 1
        q"$mods class $tpname[..$tparams] $ctorMods(...$paramss) extends { ..$earlydefns } with ..$parents { $self => ..$transformedStats }"
      }
      case q"$mods object $tname extends { ..$earlydefns } with ..$parents { $self => ..$body }" => {
        annottee // Don't fail noisly when a companion object is passed in with the actual class def
      }
      // Currently disallow on traits, this won't work well with inheritance.
      case q"$mods def $tname[..$tparams](...$paramss): $tpt = $expr" => {
        val transformedExpr = transformHierarchicalMethod(expr)
        namedElts += 1
        q"$mods def $tname[..$tparams](...$paramss): $tpt = $transformedExpr"
      }
      case other => c.abort(c.enclosingPosition, s"@chiselName annotion may only be used on classes and methods, got ${showCode(other)}")
    })

    if (namedElts != 1) {
      c.abort(c.enclosingPosition, s"@chiselName annotation did not match exactly one valid tree, got:\r\n${annottees.map(tree => showCode(tree)).mkString("\r\n\r\n")}")
    }

    q"..$transformed"
  }
}

@compileTimeOnly("enable macro paradise to expand macro annotations")
class dump extends StaticAnnotation {
  def macroTransform(annottees: Any*): Any = macro chisel3.internal.naming.DebugTransforms.dump
}
@compileTimeOnly("enable macro paradise to expand macro annotations")
class treedump extends StaticAnnotation {
  def macroTransform(annottees: Any*): Any = macro chisel3.internal.naming.DebugTransforms.treedump
}
@compileTimeOnly("enable macro paradise to expand macro annotations")
class chiselName extends StaticAnnotation {
  def macroTransform(annottees: Any*): Any = macro chisel3.internal.naming.NamingTransforms.chiselName
}

// See LICENSE for license details.

package chisel3.internal

import scala.reflect.macros.whitebox.Context
import scala.language.experimental.macros
import scala.annotation.StaticAnnotation
import scala.annotation.compileTimeOnly

// Workaround for https://github.com/sbt/sbt/issues/3966
object RuntimeDeprecatedTransform
class RuntimeDeprecatedTransform(val c: Context) {
  import c.universe._

  /** Adds a Builder.deprecated(...) call based on the contents of a plain @deprecated annotation.
    */
  def runtimeDeprecated(annottees: c.Tree*): c.Tree = {
    val transformed = annottees.map(annottee => annottee match {
      case q"$mods def $tname[..$tparams](...$paramss): $tpt = $expr" => {
        val Modifiers(_, _, annotations) = mods
        val annotationMessage = annotations.collect {  // get all messages from deprecated annotations
          case q"new deprecated($desc, $since)" => desc
        } match {  // ensure there's only one and return it
          case msg :: Nil => msg
          case _ => c.abort(c.enclosingPosition, s"@chiselRuntimeDeprecated annotion must be used with exactly one @deprecated annotation, got annotations $annotations")
        }
        val message = s"$tname is deprecated: $annotationMessage"
        val transformedExpr = q""" {
        _root_.chisel3.internal.Builder.deprecated($message)
        $expr
        } """
        q"$mods def $tname[..$tparams](...$paramss): $tpt = $transformedExpr"
      }
      case other => c.abort(c.enclosingPosition, s"@chiselRuntimeDeprecated annotion may only be used on defs, got ${showCode(other)}")
    })
    q"..$transformed"
  }
}

@compileTimeOnly("enable macro paradise to expand macro annotations")
class chiselRuntimeDeprecated extends StaticAnnotation {
  def macroTransform(annottees: Any*): Any = macro chisel3.internal.RuntimeDeprecatedTransform.runtimeDeprecated
}

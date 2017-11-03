// See LICENSE for license details.

package chisel3.core

import scala.language.experimental.macros
import scala.reflect.macros.blackbox.Context

trait CompileOptions {
  // Should Record connections require a strict match of fields.
  // If true and the same fields aren't present in both source and sink, a MissingFieldException,
  // MissingLeftFieldException, or MissingRightFieldException will be thrown.
  val connectFieldsMustMatch: Boolean
  // When creating an object that takes a type argument, the argument must be unbound (a pure type).
  val declaredTypeMustBeUnbound: Boolean
  // If a connection operator fails, don't try the connection with the operands (source and sink) reversed.
  val dontTryConnectionsSwapped: Boolean
  // If connection directionality is not explicit, do not use heuristics to attempt to determine it.
  val dontAssumeDirectionality: Boolean
  // Check that referenced Data have actually been declared.
  val checkSynthesizable: Boolean
  // Require explicit assignment of DontCare to generate "x is invalid"
  val explicitInvalidate: Boolean
}

object CompileOptions {
  // Provides a low priority Strict default. Can be overridden by importing the NotStrict option.
  // Implemented as a macro to prevent this from being used inside chisel core.
  implicit def materialize: CompileOptions = macro materialize_impl

  def materialize_impl(c: Context): c.Tree = {
    import c.universe._
    q"_root_.chisel3.core.ExplicitCompileOptions.Strict"
  }
}

object ExplicitCompileOptions {
  case class CompileOptionsClass (
                             // Should Record connections require a strict match of fields.
                             // If true and the same fields aren't present in both source and sink, a MissingFieldException,
                             // MissingLeftFieldException, or MissingRightFieldException will be thrown.
                             val connectFieldsMustMatch: Boolean,
                             // When creating an object that takes a type argument, the argument must be unbound (a pure type).
                             val declaredTypeMustBeUnbound: Boolean,
                             // If a connection operator fails, don't try the connection with the operands (source and sink) reversed.
                             val dontTryConnectionsSwapped: Boolean,
                             // If connection directionality is not explicit, do not use heuristics to attempt to determine it.
                             val dontAssumeDirectionality: Boolean,
                             // Check that referenced Data have actually been declared.
                             val checkSynthesizable: Boolean,
                             // Require an explicit DontCare assignment to generate a firrtl DefInvalid
                             val explicitInvalidate: Boolean
                           ) extends CompileOptions

  // Collection of "not strict" connection compile options.
  // These provide compatibility with existing code.
  // import chisel3.core.ExplicitCompileOptions.NotStrict
  implicit val NotStrict = new CompileOptionsClass (
    connectFieldsMustMatch = false,
    declaredTypeMustBeUnbound = false,
    dontTryConnectionsSwapped = false,
    dontAssumeDirectionality = false,
    checkSynthesizable = false,
    explicitInvalidate = false
  )

  // Collection of "strict" connection compile options, preferred for new code.
  // import chisel3.core.ExplicitCompileOptions.Strict
  implicit val Strict = new CompileOptionsClass (
    connectFieldsMustMatch = true,
    declaredTypeMustBeUnbound = true,
    dontTryConnectionsSwapped = true,
    dontAssumeDirectionality = true,
    checkSynthesizable = true,
    explicitInvalidate = true
  )
}

// See LICENSE for license details.

package chisel3

import chisel3.core.{CompileOptions, UserModule}

/** The util package provides extensions to core chisel for common hardware components and utility functions.
  *
  */
package object util {

  /** Synonyms, moved from main package object - maintain scope. */
  type ValidIO[+T <: Data] = chisel3.util.Valid[T]
  val ValidIO = chisel3.util.Valid
  val DecoupledIO = chisel3.util.Decoupled

  // A significant issue is which CompileOptions should be in effect when we elaborate utility extensions.
  // If there is a parent module with compile options, use that, otherwise, materialize the default.
  // I'd like to make this implicit, but that currently results in ambiguity with other implicit CompileOptions
  //  definitions. We should review these mechanisms and define the OneTrueWay to provide implicit CompileOptions.
  // Note: If this does become implicit, it can be overridden either by passing in explicit CompileOptions,
  //  or adding an implicit at the site where we invoke the utility class/method.
  val inheritedCompileOptions = chisel3.internal.Builder.currentModule match {
    case Some(u: UserModule) => u.compileOptions
    case _ => CompileOptions.materialize
  }
}

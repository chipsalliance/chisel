// SPDX-License-Identifier: Apache-2.0

// This file contains macros for adding source locators at the point of invocation.
//
// This is not part of coreMacros to disallow this macro from being implicitly invoked in Chisel
// frontend (and generating source locators in Chisel core), which is almost certainly a bug.
//
// Note: While these functions and definitions are not private (macros can't be
// private), these are NOT meant to be part of the public API (yet) and no
// forward compatibility guarantees are made.
// A future revision may stabilize the source locator API to allow library
// writers to append source locator information at the point of a library
// function invocation.

package chisel3.experimental

import scala.language.experimental.macros
import scala.reflect.macros.blackbox.Context
import chisel3.internal.sourceinfo.SourceInfoMacro

// Technically this should be called "SourceInfo$Intf" but that causes issues for IntelliJ so we
// omit the '$'.
private[chisel3] trait SourceInfoIntf { self: SourceInfo.type =>
  implicit def materialize: SourceInfo = macro SourceInfoMacro.generate_source_info
}

// SPDX-License-Identifier: Apache-2.0

// This file contains macros for adding source locators at the point of invocation.
//
// This is not part of coreMacros to disallow this macro from being implicitly invoked in Chisel
// frontend (and generating source locators in Chisel core), which is almost certainly a bug.

package chisel3.experimental

import chisel3.internal.sourceinfo.SourceInfoMacro

// Technically this should be called "SourceInfo$Intf" but that causes issues for IntelliJ so we
// omit the '$'.
private[chisel3] trait SourceInfoIntf { self: SourceInfo.type =>
  implicit inline def materialize: SourceInfo = ${ SourceInfoMacro.generate_source_info }
}

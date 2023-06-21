// SPDX-License-Identifier: Apache-2.0

// This file contains macros for adding source locators at the point
// of invocation.
//
// This is not part of coreMacros to disallow this macro from being
// implicitly invoked in Chisel frontend (and generating source
// locators in Chisel core), which is almost certainly a bug.
//
// As of Chisel 3.6, these methods are deprecated in favor of the
// public API in chisel3.experimental.

package chisel3.internal

package object sourceinfo

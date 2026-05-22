// SPDX-License-Identifier: Apache-2.0
package chisel3.debug

import chisel3.Data

/** Marks a [[chisel3.Data]] whose primary-constructor arguments are
  * structural and already captured by per-subfield debug emission, so
  * `DebugIntrinsics` must not attach a duplicating `params=` attribute.
  *
  * A marker trait (rather than naming the type directly) is required because
  * some such types (e.g. `chisel3.util.MixedVec`) live in the `chisel`
  * module, which depends on `core`; `core` cannot reference them by type.
  */
private[chisel3] trait SuppressDebugParams { self: Data => }

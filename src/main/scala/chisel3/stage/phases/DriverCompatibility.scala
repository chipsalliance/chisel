// SPDX-License-Identifier: Apache-2.0

package chisel3.stage.phases

import firrtl.annotations.Annotation
import firrtl.options.Phase

/** This formerly provided components of a compatibility wrapper around Chisel's removed `chisel3.Driver`.
  *
  * This object formerly included [[firrtl.options.Phase Phase]]s that generate [[firrtl.annotations.Annotation]]s
  * derived from the deprecated `firrtl.stage.phases.DriverCompatibility.TopNameAnnotation`.
  */
@deprecated("This object contains no public members. This will be removed in Chisel 3.6.", "Chisel 3.5")
object DriverCompatibility

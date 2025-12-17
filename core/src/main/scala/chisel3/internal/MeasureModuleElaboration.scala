// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

import jdk.jfr.{Event, Label, Name}

@Name("org.chipsalliance.chisel.ModuleElaboration")
@Label("Chisel Module Elaboration")
private[chisel3] class MeasureModuleElaboration extends Event {
  @Label("Module Name")
  var module: String = ""
}
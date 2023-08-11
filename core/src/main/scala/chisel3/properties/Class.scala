// SPDX-License-Identifier: Apache-2.0

package chisel3.properties

import chisel3.experimental.{BaseModule, SourceInfo}
import chisel3.internal.{throwException, Builder}
import chisel3.internal.firrtl.{Component, DefClass}

class Class extends BaseModule {
  private[chisel3] override def generateComponent(): Option[Component] = {
    Some(DefClass(this, name, Seq(), Seq()))
  }

  private[chisel3] override def initializeInParent(): Unit = ()
}

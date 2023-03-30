// SPDX-License-Identifier: Apache-2.0

package svsimTests

import svsim._

object Resources {
  implicit class TestWorkspace(workspace: Workspace) {
    def elaborateGCD(): Unit = {
      workspace.addPrimarySourceFromResource(getClass, "/GCD.sv")
      workspace.elaborate(
        ModuleInfo(
          name = "GCD",
          ports = Seq(
            new ModuleInfo.Port(
              name = "clock",
              isSettable = true,
              isGettable = true
            ),
            new ModuleInfo.Port(
              name = "a",
              isSettable = true,
              isGettable = true
            ),
            new ModuleInfo.Port(
              name = "b",
              isSettable = true,
              isGettable = true
            ),
            new ModuleInfo.Port(
              name = "loadValues",
              isSettable = true,
              isGettable = true
            ),
            new ModuleInfo.Port(
              name = "isValid",
              isSettable = false,
              isGettable = true
            ),
            new ModuleInfo.Port(
              name = "result",
              isSettable = false,
              isGettable = true
            )
          )
        )
      )
    }
  }
}

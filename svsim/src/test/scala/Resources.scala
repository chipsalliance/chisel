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
    def elaborateSIntTest(): Unit = {
      workspace.addPrimarySourceFromResource(getClass, "/SIntTest.sv")
      workspace.elaborate(
        ModuleInfo(
          name = "SIntTest",
          ports = Seq(
            new ModuleInfo.Port(
              name = "in_8",
              isSettable = true,
              isGettable = true
            ),
            new ModuleInfo.Port(
              name = "in_31",
              isSettable = true,
              isGettable = true
            ),
            new ModuleInfo.Port(
              name = "in_32",
              isSettable = true,
              isGettable = true
            ),
            new ModuleInfo.Port(
              name = "in_33",
              isSettable = true,
              isGettable = true
            ),
            new ModuleInfo.Port(
              name = "out_8",
              isSettable = false,
              isGettable = true
            ),
            new ModuleInfo.Port(
              name = "out_31",
              isSettable = false,
              isGettable = true
            ),
            new ModuleInfo.Port(
              name = "out_32",
              isSettable = false,
              isGettable = true
            ),
            new ModuleInfo.Port(
              name = "out_33",
              isSettable = false,
              isGettable = true
            ),
            new ModuleInfo.Port(
              name = "out_const_8",
              isSettable = false,
              isGettable = true
            ),
            new ModuleInfo.Port(
              name = "out_const_31",
              isSettable = false,
              isGettable = true
            ),
            new ModuleInfo.Port(
              name = "out_const_32",
              isSettable = false,
              isGettable = true
            ),
            new ModuleInfo.Port(
              name = "out_const_33",
              isSettable = false,
              isGettable = true
            )
          )
        )
      )
    }
    def elaborateInitialTest(): Unit = {
      workspace.addPrimarySourceFromResource(getClass, "/Initial.sv")
      workspace.elaborate(
        ModuleInfo(
          name = "Initial",
          ports = Seq(
            new ModuleInfo.Port(
              name = "b",
              isSettable = false,
              isGettable = true
            )
          )
        )
      )
    }
    def elaborateFinishTest(): Unit = {
      workspace.addPrimarySourceFromResource(getClass, "/Finish.sv")
      workspace.elaborate(
        ModuleInfo(
          name = "Finish",
          ports = Seq(
            new ModuleInfo.Port(
              name = "clock",
              isSettable = true,
              isGettable = false
            )
          )
        )
      )
    }
  }
}

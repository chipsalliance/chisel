// SPDX-License-Identifier: Apache-2.0

package chiselTests.stage

import chisel3._
import chisel3.testers.TestUtils
import chisel3.experimental.SourceInfo
import circt.stage.ChiselStage

import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers
import firrtl.options.OptionsException

import java.io.File

object WarningConfigurationSpec {
  class ModuleWithWarning extends RawModule {
    val in = IO(Input(UInt(8.W)))
    val out = IO(Output(UInt(8.W)))
    TestUtils.warn(1, "sample warning")
    out := in
  }

  // Examples of every existing warning
  // These will eventually become errors
  class UnsafeUIntCastToEnum extends RawModule {
    object MyEnum extends ChiselEnum {
      val a, b, c = Value
    }
    val in = IO(Input(UInt(2.W)))
    val out = IO(Output(MyEnum()))
    out := in.asTypeOf(MyEnum())
  }
  class DynamicBitSelectTooWide extends RawModule {
    val in = IO(Input(UInt(4.W)))
    val idx = IO(Input(UInt(3.W)))
    in(idx)
  }
  class DynamicBitSelectTooNarrow extends RawModule {
    val in = IO(Input(UInt(8.W)))
    val idx = IO(Input(UInt(2.W)))
    in(idx)
  }
  class DynamicIndexTooWide extends RawModule {
    val in = IO(Input(Vec(4, UInt(8.W))))
    val idx = IO(Input(UInt(3.W)))
    in(idx)
  }
  class DynamicIndexTooNarrow extends RawModule {
    val in = IO(Input(Vec(8, UInt(8.W))))
    val idx = IO(Input(UInt(2.W)))
    in(idx)
  }
  class ExtractFromVecSizeZero extends RawModule { // = Value(6)
    val in = IO(Input(Vec(0, UInt(8.W))))
    val idx = IO(Input(UInt(8.W)))
    in(idx)
  }
}

class WarningConfigurationSpec extends AnyFunSpec with Matchers with chiselTests.Utils {
  import WarningConfigurationSpec._

  private def checkInvalid(wconf: String, carat: String, expected: String): Unit = {
    val args = Array("--warn-conf", wconf)
    val e = the[OptionsException] thrownBy (ChiselStage.emitCHIRRTL(new ModuleWithWarning, args))
    val msg = e.getMessage()
    val lines = msg.split("\n")
    msg should include(expected)
    lines should contain("  " + wconf)
    lines should contain("  " + carat)
  }

  private def checkInvalid(wconf: File, badLine: String, carat: String, expected: String): Unit = {
    val args = Array("--warn-conf-file", wconf.toString)
    val e = the[OptionsException] thrownBy (ChiselStage.emitCHIRRTL(new ModuleWithWarning, args))
    val msg = e.getMessage()
    val lines = msg.split("\n")
    msg should include(expected)
    lines should contain("  " + badLine)
    lines should contain("  " + carat)
  }

  private lazy val buildDir = {
    val testRunDir = os.pwd / os.RelPath(firrtl.util.BackendCompilationUtilities.TestDirectory)
    val suiteDir = testRunDir / suiteName
    os.remove.all(suiteDir) // clear it each time
    os.makeDir.all(suiteDir)
    suiteDir
  }

  private def makeFile(name: String)(contents: String): java.io.File = {
    val file = buildDir / name
    os.write(file, contents)
    file.toIO
  }

  describe("Warning Configuration") {

    it("should support warnings as errors") {
      info("In form --warn-conf any:e")
      val args = Array("--warn-conf", "any:e", "--throw-on-first-error")
      val e = the[ChiselException] thrownBy { ChiselStage.emitCHIRRTL(new ModuleWithWarning, args) }
      e.getMessage should include("sample warning")

      info("Also in form --warnings-as-errors")
      val args2 = Array("--warnings-as-errors", "--throw-on-first-error")
      val e2 = the[ChiselException] thrownBy { ChiselStage.emitCHIRRTL(new ModuleWithWarning, args2) }
      e2.getMessage should include("sample warning")
    }

    it("should support id filters") {
      info("For suppressing warnings")
      val args = Array("--warn-conf", "id=1:s,any:e", "--throw-on-first-error")
      ChiselStage.emitCHIRRTL(new ModuleWithWarning, args)

      info("For keeping them as warnings despite --warnings-as-errors")
      val args2 = Array("--warn-conf", "id=1:w,any:e", "--throw-on-first-error")
      val (log, _) = grabLog(ChiselStage.emitCHIRRTL(new ModuleWithWarning, args2))
      log should include("sample warning")
      log should include("There were 1 warning(s) during hardware elaboration.")

      info("For elevating individual warnings to errors")
      val args3 = Array("--warn-conf", "id=1:e", "--throw-on-first-error")
      val e = the[ChiselException] thrownBy { ChiselStage.emitCHIRRTL(new ModuleWithWarning, args3) }
      e.getMessage should include("sample warning")
    }

    it("should support source filters") {
      val thisFile = implicitly[SourceInfo].filenameOption.get
      info("For suppressing warnings")
      val args = Array("--warn-conf", s"src=$thisFile:s,any:e", "--throw-on-first-error")
      ChiselStage.emitCHIRRTL(new ModuleWithWarning, args)

      info("For keeping them as warnings despite --warnings-as-errors")
      val args2 = Array("--warn-conf", s"src=$thisFile:w,any:e", "--throw-on-first-error")
      val (log, _) = grabLog(ChiselStage.emitCHIRRTL(new ModuleWithWarning, args2))
      log should include("sample warning")
      log should include("There were 1 warning(s) during hardware elaboration.")

      info("For elevating individual warnings to errors")
      val args3 = Array("--warn-conf", s"src=$thisFile:e", "--throw-on-first-error")
      val e = the[ChiselException] thrownBy { ChiselStage.emitCHIRRTL(new ModuleWithWarning, args3) }
      e.getMessage should include("sample warning")
    }

    it("should support source filter globs") {
      info("as simple extension matches")
      val args = Array("--warn-conf", "src=**.scala:s,any:e", "--throw-on-first-error")
      ChiselStage.emitCHIRRTL(new ModuleWithWarning, args)

      info("including some intermediate directories")
      val args2 = Array("--warn-conf", "src=**/stage/WarningConfigurationSpec.scala:s,any:e", "--throw-on-first-error")
      ChiselStage.emitCHIRRTL(new ModuleWithWarning, args2)

      info("including when rooted")
      val args3 =
        Array("--warn-conf", "src=src/test/scala/**/WarningConfigurationSpec.scala:s,any:e", "--throw-on-first-error")
      ChiselStage.emitCHIRRTL(new ModuleWithWarning, args3)
    }

    it("should support being specified multiple times") {
      val suppress = Array("--warn-conf", "any:s")
      val error = Array("--warn-conf", "any:e")
      val extra = Array("--throw-on-first-error")
      val (log, _) = grabLog(ChiselStage.emitCHIRRTL(new ModuleWithWarning, suppress ++ error ++ extra))
      log should be("")
      val e = the[ChiselException] thrownBy ChiselStage.emitCHIRRTL(new ModuleWithWarning, error ++ suppress ++ extra)
      e.getMessage should include("sample warning")
    }

    it("should error on a missing action") {
      val wconf = "potato"
      val carat = "     ^"
      val msg = "Filter 'potato' is missing an action"
      checkInvalid(wconf, carat, msg)
    }

    it("should error on an invalid action") {
      val wconf = "id=1:x"
      val carat = "    ^"
      val msg = "Invalid action ':x'"
      checkInvalid(wconf, carat, msg)

      info("and be able to point to an invalid action in a later filter")
      val wconf2 = "id=1:s,src=**.scala:x"
      val carat2 = "                   ^"
      val msg2 = "Invalid action ':x'"
      checkInvalid(wconf2, carat2, msg2)
    }

    it("should error on an invalid category") {
      val wconf = "id=1:s,id=2&foo=sample warning:e"
      val carat = "            ^"
      val msg = "Invalid category 'foo'"
      checkInvalid(wconf, carat, msg)
    }

    it("should error on a duplicate id") {
      val wconf = "id=1:s,id=2&id=3:e"
      val carat = "            ^"
      val msg = "Cannot have duplicates of the same category"
      checkInvalid(wconf, carat, msg)
    }

    it("should error on a duplicate src") {
      val wconf = "id=1:s,src=hi&src=bye:e"
      val carat = "              ^"
      val msg = "Cannot have duplicates of the same category"
      checkInvalid(wconf, carat, msg)
    }

    it("should error when combining any other category with 'any'") {
      val wconf = "id=1:s,any&src=bye:e"
      val carat = "           ^"
      val msg = "'any' cannot be combined with other filters"
      checkInvalid(wconf, carat, msg)

      info("In any order with any")
      val wconf2 = "id=1:s,src=bye&any:e"
      val carat2 = "               ^"
      val msg2 = "'any' cannot be combined with other filters"
      checkInvalid(wconf2, carat2, msg2)

      info("Even multiple 'any'")
      val wconf3 = "id=1:s,any&any:e"
      val carat3 = "           ^"
      val msg3 = "'any' cannot be combined with other filters"
      checkInvalid(wconf3, carat3, msg3)
    }
  }

  describe("Warning Configuration File") {

    it("should support filters") {
      info("For suppressing warnings")
      val file = makeFile("basic_suppressing.conf")(
        """|id=1:s
           |any:e
           |""".stripMargin
      )
      val args = Array("--warn-conf-file", file.toString, "--throw-on-first-error")
      ChiselStage.emitCHIRRTL(new ModuleWithWarning, args)

      info("Including source filters")
      val file2 = makeFile("with_source_filter.conf")(
        """|id=1&src=**/WarningConfigurationSpec.scala:s
           |any:e""".stripMargin
      )
      val args2 = Array("--warn-conf-file", file2.toString, "--throw-on-first-error")
      ChiselStage.emitCHIRRTL(new ModuleWithWarning, args2)
    }

    it("should support line comments") {
      val file = makeFile("with_comments.conf")(
        """|# Here is a comment
           |id=1:s
           |# And another one!
           |any:e
           |""".stripMargin
      )
      val args = Array("--warn-conf-file", file.toString, "--throw-on-first-error")
      ChiselStage.emitCHIRRTL(new ModuleWithWarning, args)
    }

    it("should have good error messages") {
      info("For invalid actions")
      val badln = "id=1:x"
      val carat = "    ^"
      val badAction = makeFile("bad_action.conf")(
        s"""|$badln
            |any:e""".stripMargin
      )
      checkInvalid(badAction, badln, carat, "Invalid action ':x'")

      info("Including across multiple lines")
      val badln2 = "id=1&id=3:e"
      val carat2 = "     ^"
      val badAction2 = makeFile("bad_action2.conf")(
        s"""|# How about a comment?
            |id=4&src=**.scala:s
            |$badln2
            |any:w""".stripMargin
      )
      checkInvalid(badAction2, badln2, carat2, "Cannot have duplicates of the same category")
    }

    it("should work when specified multiple times") {
      val suppressConf = makeFile("supress_all.conf")(
        """|any:s
           |""".stripMargin
      )
      val errorConf = makeFile("error_all.conf")(
        """|any:e
           |""".stripMargin
      )
      val suppress = Array("--warn-conf-file", suppressConf.toString)
      val errorFile = Array("--warn-conf-file", errorConf.toString)
      val errorArgs = Array("--warn-conf", "any:e")
      val extra = Array("--throw-on-first-error")

      info("For suppressing")
      val (log, _) = grabLog(ChiselStage.emitCHIRRTL(new ModuleWithWarning, suppress ++ errorFile ++ extra))
      log should be("")

      info("For erroring")
      val e =
        the[ChiselException] thrownBy ChiselStage.emitCHIRRTL(new ModuleWithWarning, errorFile ++ suppress ++ extra)
      e.getMessage should include("sample warning")

      info("Also when composed with --warn-conf")
      val e2 =
        the[ChiselException] thrownBy ChiselStage.emitCHIRRTL(new ModuleWithWarning, errorArgs ++ suppress ++ extra)
      e2.getMessage should include("sample warning")
    }
  }

  // Important to test the specific numbering of existing warnings to guard against accidental changes
  describe("Warning Configuration Numbering") {

    it("should number UnsafeUIntCastToEnum as 1") {
      val args = Array("--warn-conf", "id=1:e,any:s", "--throw-on-first-error")
      val e = the[Exception] thrownBy ChiselStage.emitCHIRRTL(new UnsafeUIntCastToEnum, args)
      (e.getMessage should include).regex("""\[W001\] Casting non-literal UInt to.*MyEnum""")
    }

    it("should number DynamicBitSelectTooWide as 2") {
      val args = Array("--warn-conf", "id=2:e,any:s", "--throw-on-first-error")
      val e = the[Exception] thrownBy ChiselStage.emitCHIRRTL(new DynamicBitSelectTooWide, args)
      e.getMessage should include("[W002] Dynamic index with width 3 is too large for extractee of width 4")
    }

    it("should number DynamicBitSelectTooNarrow as 3") {
      val args = Array("--warn-conf", "id=3:e,any:s", "--throw-on-first-error")
      val e = the[Exception] thrownBy ChiselStage.emitCHIRRTL(new DynamicBitSelectTooNarrow, args)
      e.getMessage should include("[W003] Dynamic index with width 2 is too small for extractee of width 8")
    }

    it("should number DynamicIndexTooWide as 4") {
      val args = Array("--warn-conf", "id=4:e,any:s", "--throw-on-first-error")
      val e = the[Exception] thrownBy ChiselStage.emitCHIRRTL(new DynamicIndexTooWide, args)
      e.getMessage should include("[W004] Dynamic index with width 3 is too wide for Vec of size 4")
    }

    it("should number DynamicIndexTooNarrow as 5") {
      val args = Array("--warn-conf", "id=5:e,any:s", "--throw-on-first-error")
      val e = the[Exception] thrownBy ChiselStage.emitCHIRRTL(new DynamicIndexTooNarrow, args)
      e.getMessage should include("[W005] Dynamic index with width 2 is too narrow for Vec of size 8")
    }

    it("should number ExtractFromVecSizeZero as 6") {
      val args = Array("--warn-conf", "id=6:e,any:s", "--throw-on-first-error")
      val e = the[Exception] thrownBy ChiselStage.emitCHIRRTL(new ExtractFromVecSizeZero, args)
      e.getMessage should include("[W006] Cannot extra from Vec of size 0")
    }
  }
}

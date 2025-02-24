// SPDX-License-Identifier: Apache-2.0

package chisel3.simulator.scalatest

import chisel3.simulator.HasTestingDirectory
import java.nio.file.{FileSystems, Path}
import org.scalatest.TestSuite
import scala.util.DynamicVariable

/** A mix-in for a Scalatest test suite that will setup the output directory for
  * you.  E.g., this will create output directories for your tests like the
  * following:
  *
  * {{{
  * <buildDir>
  * └── <suite-name>
  *     └── <scope-1-name>
  *         └── ...
  *             └── <scope-n-name>
  *                 ├── <test-1-name>
  *                 ├── ...
  *                 └── <test-n-name>
  * }}}
  *
  * You may change the `buildDir` by overridding a method of the same name in
  * this trait.
  *
  */
trait WithTestingDirectory { self: TestSuite =>

  /** Return the name of the root test directory.
    *
    * For different behavior, please override this in your test suite.
    */
  def buildDir: String = "build"

  // Assemble all the directories that should be created for this test.  This is
  // done by examining the test (via a fixture) and then setting a dynamic
  // variable for that test.  The exacct structure of the test is extracted,
  // including any nesting (Scalatest scopes) in the test.
  private val testName: DynamicVariable[List[String]] = new DynamicVariable[List[String]](Nil)
  override def withFixture(test: NoArgTest) = {
    testName.withValue(test.scopes.toList :+ test.text) {
      test()
    }
  }

  /** Implementaton of [[HasTestingDirectory]] which sets up the test directory
    * for you based on settings which make sense in Scalatest.
    */
  final implicit def implementation: HasTestingDirectory = new HasTestingDirectory {

    // A sequence of regular expressions and their replacements that should be
    // applied to the test name.
    val res = Seq("\\s|\\(|\\)|\\$".r -> "-", "\"|\'|#".r -> "")

    /** Return the test name with minimal sanitization applied:
      *
      *   - Replace all whitespace as this is incompatible with GNU make [1]
      *   - Replace any characters which Verilator Makefiles empirically have
      *     problems with
      *
      * [1]: https://savannah.gnu.org/bugs/?712
      */
    final def getTestName = testName.value.map { case a =>
      res.foldLeft(a) { case (string, (regex, replacement)) => regex.replaceAllIn(string, replacement) }
    }

    override def getDirectory: Path = FileSystems
      .getDefault()
      .getPath(buildDir, self.suiteName +: getTestName: _*)

  }

}

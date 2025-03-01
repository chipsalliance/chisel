// SPDX-License-Identifier: Apache-2.0

package chisel3.testing

import chisel3.testing.scalatest.TestingDirectory
import org.scalatest.TestSuite

package object scalatest {

  /** A trait that provides FileCheck APIs and integration with Scalatest.
    *
    * Example usage:
    * {{{
    * import chisel3.testing.scalatest.FileCheck.scalatest
    * import org.scalatest.flatspec.AnyFlatSpec
    * import org.scalatest.matches.should.Matchers
    *
    * class Foo extends AnyFlatSpec with Matchers with FileCheck {
    *   /** This has access to all FileCheck APIs like `fileCheck`. */
    * }
    * }}}
    *
    * @see [[chisel3.testing.FileCheck]]
    */
  trait FileCheck extends chisel3.testing.FileCheck with TestingDirectory { self: TestSuite => }

}

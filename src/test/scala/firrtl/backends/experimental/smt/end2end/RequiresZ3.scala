// See LICENSE for license details.

package firrtl.backends.experimental.smt.end2end

import org.scalatest.Tag

// To disable tests that require the Z3 SMT solver to be installed use the following:
//    `sbt testOnly -- -l RequiresZ3`
object RequiresZ3 extends Tag("RequiresZ3")

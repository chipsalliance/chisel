// See LICENSE for license details.

package tags

import org.scalatest.Tag

// To disable tests tagged as `TagRequiresSimulator`, use the following:
//    `sbt testOnly -- -l TagRequiresSimulator`
// Note: the string specified with the `-l`, is the String argument passed to the `Tag` constructor.
object TagRequiresSimulator extends Tag("TagRequiresSimulator")

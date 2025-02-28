// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import firrtl.FileUtils
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import scala.annotation.nowarn

<<<<<<< HEAD
||||||| parent of 868c1958 (Mass deprecations, to be removed in Chisel 7 (#4754))
import java.nio.file.Files
import java.nio.file.StandardCopyOption.REPLACE_EXISTING

=======
import java.nio.file.Files
import java.nio.file.StandardCopyOption.REPLACE_EXISTING

@nowarn("msg=object FileUtils in package firrtl is deprecated")
>>>>>>> 868c1958 (Mass deprecations, to be removed in Chisel 7 (#4754))
class FileUtilsSpec extends AnyFlatSpec with Matchers {

  private val sampleAnnotations:         String = "annotations/SampleAnnotations.anno.json"
  private val sampleAnnotationsFileName: String = s"src/test/resources/$sampleAnnotations"

  behavior.of("FileUtils.getLines")

  it should "read from a Java file" in {
    FileUtils.getLines(sampleAnnotationsFileName).size should be > 0
  }

  behavior.of("FileUtils.getText")

  it should "read from a string filename" in {
    FileUtils.getText(sampleAnnotationsFileName).size should be > 0
  }

}

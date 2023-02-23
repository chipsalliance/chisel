// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import firrtl.FileUtils
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

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

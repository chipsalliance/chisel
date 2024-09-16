// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import firrtl.FileUtils
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import java.nio.file.Files
import java.nio.file.StandardCopyOption.REPLACE_EXISTING

class FileUtilsSpec extends AnyFlatSpec with Matchers {

  private val sampleAnnotations: String = "/annotations/SampleAnnotations.anno.json"

  def withSampleAnnotations(f: String => Unit): Unit = {
    val tempFile = Files.createTempFile("SampleAnnotations", ".anno.json")
    val source = getClass.getResourceAsStream(sampleAnnotations)
    Files.copy(source, tempFile, REPLACE_EXISTING)
    f(tempFile.toString)
    Files.delete(tempFile)
  }

  behavior.of("FileUtils.getLines")

  it should "read from a Java file" in withSampleAnnotations { sampleAnnotationsFileName =>
    FileUtils.getLines(sampleAnnotationsFileName).size should be > 0
  }

  behavior.of("FileUtils.getText")

  it should "read from a string filename" in withSampleAnnotations { sampleAnnotationsFileName =>
    FileUtils.getText(sampleAnnotationsFileName).size should be > 0
  }

}

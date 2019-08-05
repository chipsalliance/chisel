// See LICENSE for license details.

package firrtlTests

import org.scalatest.{FlatSpec, Matchers}

import firrtl.FileUtils

class FileUtilsSpec extends FlatSpec with Matchers {

  private val sampleAnnotations: String = "annotations/SampleAnnotations.anno"
  private val sampleAnnotationsFileName: String = s"src/test/resources/$sampleAnnotations"

  behavior of "FileUtils.getLines"

  it should "read from a string filename" in {
    FileUtils.getLines(sampleAnnotationsFileName).size should be > 0
  }

  it should "read from a Java file" in {
    FileUtils.getLines(new java.io.File(sampleAnnotationsFileName)).size should be > 0
  }

  behavior of "FileUtils.getText"

  it should "read from a string filename" in {
    FileUtils.getText(sampleAnnotationsFileName).size should be > 0
  }

  it should "read from a Java file" in {
    FileUtils.getText(new java.io.File(sampleAnnotationsFileName)).size should be > 0
  }

  behavior of "FileUtils.getLinesResource"

  it should "read from a resource" in {
    FileUtils.getLinesResource(s"/$sampleAnnotations").size should be > 0
  }

  behavior of "FileUtils.getTextResource"

  it should "read from a resource" in {
    FileUtils.getTextResource(s"/$sampleAnnotations").split("\n").size should be > 0
  }

}

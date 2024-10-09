// SPDX-License-Identifier: Apache-2.0

package firrtlTests.annotationTests

import firrtl.AttributeAnnotation
import firrtl.annotations._
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers
import org.json4s.convertToJsonInput

class AttributeAnnotationSpec extends AnyFreeSpec with Matchers {
  "AttributeAnnotation should be correctly parsed from a string" in {
    val attribAnno = new AttributeAnnotation(
      ComponentName("attrib", ModuleName("ModuleAttrib", CircuitName("CircuitAttrib"))),
      "X_INTERFACE_INFO = \"some:interface:type:1.0 SIGNAL\""
    )

    val annoString = JsonProtocol.serializeTry(Seq(attribAnno)).get
    val loadedAnnos = JsonProtocol.deserializeTry(annoString).get
    attribAnno should equal(loadedAnnos.head)
  }
}

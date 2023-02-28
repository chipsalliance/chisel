// SPDX-License-Identifier: Apache-2.0

package firrtlTests.annotationTests

import firrtl.RenameMap
import firrtl.annotations._
import firrtl.testutils.FirrtlFlatSpec

object AnnotationSpec {
  case class TestAnno(pairs: List[(String, ReferenceTarget)]) extends Annotation {
    def update(renames: RenameMap): Seq[Annotation] = {
      val pairsx = pairs.flatMap {
        case (n, t) =>
          val ts = renames
            .get(t)
            .map(_.map(_.asInstanceOf[ReferenceTarget]))
            .getOrElse(Seq(t))
          ts.map(n -> _)
      }
      Seq(TestAnno(pairsx))
    }
  }
}

class AnnotationSpec extends FirrtlFlatSpec {
  import AnnotationSpec._

  behavior.of("Annotation.getTargets")

  it should "not stack overflow" in {
    val ref = CircuitTarget("Top").module("Foo").ref("vec")
    val anno = TestAnno((0 until 10000).map(i => (i.toString, ref.index(i))).toList)
    anno.getTargets should be(anno.pairs.map(_._2))
  }
}

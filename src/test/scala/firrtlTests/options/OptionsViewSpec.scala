// SPDX-License-Identifier: Apache-2.0

package firrtlTests.options

import firrtl.options.OptionsView
import firrtl.AnnotationSeq
import firrtl.annotations.{Annotation, NoTargetAnnotation}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class OptionsViewSpec extends AnyFlatSpec with Matchers {

  /* Annotations */
  case class NameAnnotation(name: String) extends NoTargetAnnotation
  case class ValueAnnotation(value: Int) extends NoTargetAnnotation

  /* The type we want to view the annotations as */
  case class Foo(name: Option[String] = None, value: Option[Int] = None)
  case class Bar(name: String = "bar")

  /* An OptionsView that converts an AnnotationSeq to Option[Foo] */
  implicit object FooView extends OptionsView[Foo] {
    private def append(foo: Foo, anno: Annotation): Foo = anno match {
      case NameAnnotation(n)  => foo.copy(name = Some(n))
      case ValueAnnotation(v) => foo.copy(value = Some(v))
      case _                  => foo
    }

    def view(options: AnnotationSeq): Foo = options.foldLeft(Foo())(append)
  }

  /* An OptionsView that converts an AnnotationSeq to Option[Bar] */
  implicit object BarView extends OptionsView[Bar] {
    private def append(bar: Bar, anno: Annotation): Bar = anno match {
      case NameAnnotation(n) => bar.copy(name = n)
      case _                 => bar
    }

    def view(options: AnnotationSeq): Bar = options.foldLeft(Bar())(append)
  }

  behavior.of("OptionsView")

  it should "convert annotations to one of two types" in {
    /* Some default annotations */
    val annos = Seq(NameAnnotation("foo"), ValueAnnotation(42))

    info("Foo conversion okay")
    FooView.view(annos) should be(Foo(Some("foo"), Some(42)))

    info("Bar conversion okay")
    BarView.view(annos) should be(Bar("foo"))
  }

  behavior.of("Viewer")

  it should "implicitly view annotations as the specified type" in {
    import firrtl.options.Viewer._

    /* Some empty annotations */
    val annos = Seq[Annotation]()

    info("Foo view okay")
    view[Foo](annos) should be(Foo(None, None))

    info("Bar view okay")
    view[Bar](annos) should be(Bar())
  }
}

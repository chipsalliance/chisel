// See LICENSE for license details

package firrtlTests.options

import org.scalatest.{FlatSpec, Matchers}

import firrtl.options.OptionsView
import firrtl.AnnotationSeq
import firrtl.annotations.{Annotation,NoTargetAnnotation}

class OptionsViewSpec extends FlatSpec with Matchers {

  /* Annotations */
  case class NameAnnotation(name: String) extends NoTargetAnnotation
  case class ValueAnnotation(value: Int) extends NoTargetAnnotation

  /* The type we want to view the annotations as */
  case class Foo(name: Option[String] = None, value: Option[Int] = None)
  case class Bar(name: String = "bar")

  /* An OptionsView that converts an AnnotationSeq to Option[Foo] */
  implicit object FooView extends OptionsView[Foo] {
    private def append(foo: Foo, anno: Annotation): Foo = anno match {
      case NameAnnotation(n)  => foo.copy(name  = Some(n))
      case ValueAnnotation(v) => foo.copy(value = Some(v))
      case _                  => foo
    }

    def view(options: AnnotationSeq): Option[Foo] = {
      val annoSeq = options.foldLeft(Foo())(append)
      Some(annoSeq)
    }
  }

  /* An OptionsView that converts an AnnotationSeq to Option[Bar] */
  implicit object BarView extends OptionsView[Bar] {
    private def append(bar: Bar, anno: Annotation): Bar = anno match {
      case NameAnnotation(n) => bar.copy(name = n)
      case _                 => bar
    }

    def view(options: AnnotationSeq): Option[Bar] = {
      val annoSeq = options.foldLeft(Bar())(append)
      Some(annoSeq)
    }
  }

  behavior of "OptionsView"

  it should "convert annotations to one of two types" in {
    /* Some default annotations */
    val annos = Seq(NameAnnotation("foo"), ValueAnnotation(42))

    info("Foo conversion okay")
    FooView.view(annos) should be (Some(Foo(Some("foo"), Some(42))))

    info("Bar conversion okay")
    BarView.view(annos) should be (Some(Bar("foo")))
  }

  behavior of "Viewer"

  it should "implicitly view annotations as the specified type" in {
    import firrtl.options.Viewer._

    /* Some empty annotations */
    val annos = Seq[Annotation]()

    info("Foo view okay")
    view[Foo](annos) should be (Some(Foo(None, None)))

    info("Bar view okay")
    view[Bar](annos) should be (Some(Bar()))
  }
}

// See LICENSE for license details.

package firrtlTests.options

import org.scalatest.{FlatSpec, Matchers}
import scopt.OptionParser
import java.util.ServiceLoader

import firrtl.options.{RegisteredTransform, RegisteredLibrary}
import firrtl.passes.Pass
import firrtl.ir.Circuit
import firrtl.annotations.NoTargetAnnotation
import firrtl.AnnotationSeq

case object HelloAnnotation extends NoTargetAnnotation

class FooTransform extends Pass with RegisteredTransform {
  def run(c: Circuit): Circuit = c
  def addOptions(p: OptionParser[AnnotationSeq]): Unit =
    p.opt[Unit]("hello")
      .action( (_, c) => HelloAnnotation +: c )
}

class BarLibrary extends RegisteredLibrary {
  def name: String = "Bar"
  def addOptions(p: OptionParser[AnnotationSeq]): Unit =
    p.opt[Unit]("world")
      .action( (_, c) => HelloAnnotation +: c )
}

class RegistrationSpec extends FlatSpec with Matchers {

  behavior of "RegisteredTransform"

  it should "FooTransform should be discovered by Java.util.ServiceLoader" in {
    val iter = ServiceLoader.load(classOf[RegisteredTransform]).iterator()
    val transforms = scala.collection.mutable.ArrayBuffer[RegisteredTransform]()
    while (iter.hasNext) {
      transforms += iter.next()
    }
    transforms.map(_.getClass.getName) should contain ("firrtlTests.options.FooTransform")
  }

  behavior of "RegisteredLibrary"

  it should "BarLibrary be discovered by Java.util.ServiceLoader" in {
    val iter = ServiceLoader.load(classOf[RegisteredLibrary]).iterator()
    val transforms = scala.collection.mutable.ArrayBuffer[RegisteredLibrary]()
    while (iter.hasNext) {
      transforms += iter.next()
    }
    transforms.map(_.getClass.getName) should contain ("firrtlTests.options.BarLibrary")
  }
}

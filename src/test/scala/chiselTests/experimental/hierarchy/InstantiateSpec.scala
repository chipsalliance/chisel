// SPDX-License-Identifier: Apache-2.0

package chiselTests
package experimental.hierarchy

import chisel3._
import chisel3.util.Valid
import chisel3.experimental.hierarchy._
import circt.stage.ChiselStage.convert
import chisel3.experimental.{ExtModule, IntrinsicModule}

// Note, the instantiable classes must not be inner classes because the materialized WeakTypeTags
// will be different and they will not give the same hashCode when looking up the Definition in the
// cache
object InstantiateSpec {
  // Extended in tests
  abstract class Top extends Module {
    override def desiredName = "Top"
  }

  @instantiable
  class NoArgs extends Module {
    @public val in = IO(Input(UInt(8.W)))
    @public val out = IO(Output(UInt(8.W)))
    out := in + 1.U
  }

  @instantiable
  class OneImplicitArg(implicit n: Int) extends Module {
    @public val in = IO(Input(UInt(8.W)))
    @public val out = IO(Output(UInt(8.W)))
    out := in + n.U
  }

  @instantiable
  class TwoImplicitArgs(implicit n: Int, m: String) extends Module {
    @public val in = IO(Input(UInt(8.W)))
    @public val out = IO(Output(UInt(8.W)))
    out := in + (n + m.toInt).U
  }

  @instantiable
  class OneArg(n: Int) extends Module {
    @public val in = IO(Input(UInt(8.W)))
    @public val out = IO(Output(UInt(8.W)))
    out := in + n.U
  }

  @instantiable
  class ThreeArgs(n: Int, m: Int, o: String) extends Module {
    @public val in = IO(Input(UInt(8.W)))
    @public val out = IO(Output(UInt(8.W)))
    out := in + (n + m + o.toInt).U
  }

  @instantiable
  class TupleArg(n: (Int, Int)) extends Module {
    @public val in = IO(Input(UInt(8.W)))
    @public val out = IO(Output(UInt(8.W)))
    out := in + (n._1 + n._2).U
  }

  @instantiable
  class DefaultArguments(n: Int = 10, m: Int = 11) extends Module {
    @public val in = IO(Input(UInt(8.W)))
    @public val out = IO(Output(UInt(8.W)))
    out := in + (n + m).U
  }

  @instantiable
  class MixedDefaultArguments(n: Int, m: Int = 2) extends Module {
    @public val in = IO(Input(UInt(8.W)))
    @public val out = IO(Output(UInt(8.W)))
    out := in + (n + m).U
  }

  @instantiable
  class HogWild()(n: Int)(m: Int, o: String) extends Module {
    @public val in = IO(Input(UInt(8.W)))
    @public val out = IO(Output(UInt(8.W)))
    out := in + (n + m + o.toInt).U
  }

  @instantiable
  class TypeParameterized[A](arg: A) extends Module {
    @public val in = IO(Input(UInt(8.W)))
    @public val out = IO(Output(UInt(8.W)))
    val n = arg match {
      case s: String => s.toInt
      case i: Int    => i
    }
    out := in + n.U
  }

  // TODO we should provide a better version of this in Chisel
  def dataToString(data: Data): String = data match {
    case elt: Element => elt.toString
    case vec: Vec[_]  => s"Vec_${vec.size}_${dataToString(vec(0))}"
    case rec: Record  => rec.toString
  }

  @instantiable
  class DataTypeParameterized[T <: Data](gen: T) extends Module {
    // Need to override desiredName to work around the fact that the Instantiate cache doesn't get
    // cleared between tests
    override def desiredName = s"${this.getClass.getSimpleName}_${dataToString(gen)}"
    @public val in = IO(Input(gen))
    @public val out = IO(Output(gen))
    out := in
  }

  class DataTypeParameterizedByName[T <: Data](gen: => T) extends Module {
    // Need to override desiredName to work around the fact that the Instantiate cache doesn't get
    // cleared between tests
    override def desiredName = s"${this.getClass.getSimpleName}_${dataToString(gen)}"
    @public val in = IO(Input(gen))
    @public val out = IO(Output(gen))
    out := in
  }

  sealed trait MyEnumeration
  case object FooEnum extends MyEnumeration
  case object BarEnum extends MyEnumeration
  case class FizzEnum(value: Int) extends MyEnumeration
  case class BuzzEnum(value: Int) extends MyEnumeration

  class ModuleParameterizedByProductTypes(param: MyEnumeration) extends Module {
    override def desiredName = s"${this.getClass.getSimpleName}_$param"
    val gen = param match {
      case FooEnum     => UInt(8.W)
      case BarEnum     => SInt(8.W)
      case FizzEnum(n) => Vec(n, UInt(8.W))
      case BuzzEnum(n) => Vec(n, SInt(8.W))
    }
    @public val in = IO(Input(gen))
    @public val out = IO(Output(gen))
    out := in
  }

  class ModuleParameterizedBySeq(param: Seq[Int]) extends Module {
    override def desiredName = s"${this.getClass.getSimpleName}_" + param.mkString("_")
    @public val in = param.map(w => IO(Input(UInt(w.W))))
    @public val out = param.map(w => IO(Output(UInt(w.W))))
    out.zip(in).foreach { case (o, i) => o := i }
  }

  @instantiable
  class InstantiableBlackBox extends BlackBox {
    @public val io = IO(new Bundle {
      val in = Input(UInt(8.W))
      val out = Output(UInt(8.W))
    })
  }

  @instantiable
  class InstantiableExtModule extends ExtModule {
    @public val in = IO(Input(UInt(8.W)))
    @public val out = IO(Output(UInt(8.W)))
  }

  @instantiable
  class InstantiableIntrinsic extends IntrinsicModule("MyIntrinsic", Map()) {
    @public val in = IO(Input(UInt(8.W)))
    @public val out = IO(Output(UInt(8.W)))
  }
}

class ParameterizedReset(hasAsyncNotSyncReset: Boolean) extends Module {
  override def resetType = if (hasAsyncNotSyncReset) Module.ResetType.Asynchronous else Module.ResetType.Synchronous
}

class InstantiateSpec extends ChiselFunSpec with Utils {
  import InstantiateSpec._

  describe("Module classes that take no arguments") {
    it("should be Instantiate-able") {
      val modules = convert(new Top {
        val inst0 = Instantiate(new NoArgs)
        val inst1 = Instantiate(new NoArgs)
      }).modules.map(_.name)
      assert(modules == Seq("NoArgs", "Top"))
    }
  }

  describe("Module classes that take only implicit arguments") {
    it("should be Instantiate-able if there are only a single implicit argument") {
      val modules = convert(new Top {
        implicit val n = 3
        val inst0 = Instantiate(new OneImplicitArg)
        val inst1 = Instantiate(new OneImplicitArg)
      }).modules.map(_.name)
      assert(modules == Seq("OneImplicitArg", "Top"))
    }

    it("should be Instantiate-able if there are multiple implicit arguments") {
      val modules = convert(new Top {
        implicit val n = 3
        implicit val str = "4"
        val inst0 = Instantiate(new TwoImplicitArgs)
        val inst1 = Instantiate(new TwoImplicitArgs)
      }).modules.map(_.name)
      assert(modules == Seq("TwoImplicitArgs", "Top"))
    }

    it("should be Instantiate-able when arguments are passed manually") {
      val modules = convert(new Top {
        implicit val n = 5
        implicit val str = "6"
        val inst0 = Instantiate(new TwoImplicitArgs)
        val inst1 = Instantiate(new TwoImplicitArgs()(n, str))
      }).modules.map(_.name)
      assert(modules == Seq("TwoImplicitArgs", "Top"))
    }
  }

  describe("Module classes that take a single argument list") {
    it("should be Instantiate-able when there is only a single argument") {
      val modules = convert(new Top {
        val n = 3
        val inst0 = Instantiate(new OneArg(3))
        val inst1 = Instantiate(new OneArg(n))
      }).modules.map(_.name)
      assert(modules == Seq("OneArg", "Top"))
    }

    it("should be Instantiate-able when there are 3 arguments") {
      val modules = convert(new Top {
        val inst0 = Instantiate(new ThreeArgs(3, 4, "5"))
        val inst1 = Instantiate(new ThreeArgs(3, 4, "5"))
      }).modules.map(_.name)
      assert(modules == Seq("ThreeArgs", "Top"))
    }
  }

  describe("Module classes that take default argument lists") {
    it("should be Instantiable-able with all arguments as defaults") {
      val modules = convert(new Top {
        val inst0 = Instantiate(new DefaultArguments)
        val inst1 = Instantiate(new DefaultArguments)
      }).modules.map(_.name)
      assert(modules == Seq("DefaultArguments", "Top"))
    }

    it("should be Instantiable-able with arguments passed explicitly") {
      val m = 13
      val modules = convert(new Top {
        val inst0 = Instantiate(new DefaultArguments(10, 13))
        val inst1 = Instantiate(new DefaultArguments(10, m))
      }).modules.map(_.name)
      assert(modules == Seq("DefaultArguments", "Top"))
    }

    it("should be Instantiable-able with only some default arguments passed explicitly") {
      val n = 11
      val modules = convert(new Top {
        val inst0 = Instantiate(new DefaultArguments(n))
        val inst1 = Instantiate(new DefaultArguments(11))
      }).modules.map(_.name)
      assert(modules == Seq("DefaultArguments", "Top"))
    }

    it("should be Instantiable-able with mixed regular and default arguments") {
      val n = 7
      val modules = convert(new Top {
        val inst0 = Instantiate(new MixedDefaultArguments(7))
        val inst1 = Instantiate(new MixedDefaultArguments(n, 2))
      }).modules.map(_.name)
      assert(modules == Seq("MixedDefaultArguments", "Top"))
    }

    it("should NOT compile with named arguments") {
      """
      val m = 14
      val modules = convert(new Top {
        val inst0 = Instantiate(new DefaultArguments(n = 10, m = 14))
        val inst1 = Instantiate(new DefaultArguments(m = m))
      }).modules.map(_.name)
      assert(modules == Seq("DefaultArguments", "Top"))
      """ shouldNot compile
    }
  }

  describe("Module classes that take multiple parameter lists") {
    it("should be Instantiate-able with a crazy collection of argument lists") {
      val n = 7
      val m = 18
      val s = "3"
      val modules = convert(new Top {
        val inst0 = Instantiate(new HogWild()(7)(18, "3"))
        val inst1 = Instantiate(new HogWild()(n)(m, s))
      }).modules.map(_.name)
      assert(modules == Seq("HogWild", "Top"))
    }
  }

  describe("Module classes with type parameters") {
    it("should work for non-Data type parameters") {
      val modules = convert(new Top {
        val n = "17"
        val inst0 = Instantiate(new TypeParameterized("17"))
        val inst1 = Instantiate(new TypeParameterized(n))
      }).modules.map(_.name)
      assert(modules == Seq("TypeParameterized", "Top"))
    }

    it("should work for UInt type parameters") {
      val modules = convert(new Top {
        val inst0 = Instantiate(new DataTypeParameterized(UInt(8.W)))
        val inst1 = Instantiate(new DataTypeParameterized(UInt(8.W)))
      }).modules.map(_.name)
      assert(modules == Seq("DataTypeParameterized_UInt8", "Top"))
    }

    it("should work for Vec type parameters") {
      val modules = convert(new Top {
        val inst0 = Instantiate(new DataTypeParameterized(Vec(2, UInt(8.W))))
        val inst1 = Instantiate(new DataTypeParameterized(Vec(2, UInt(8.W))))
      }).modules.map(_.name)
      assert(modules == Seq("DataTypeParameterized_Vec_2_UInt8", "Top"))
    }

    it("should work for Bundle type parameters") {
      val modules = convert(new Top {
        val inst0 = Instantiate(new DataTypeParameterized(Valid(UInt(8.W))))
        val inst1 = Instantiate(new DataTypeParameterized(Valid(UInt(8.W))))
      }).modules.map(_.name)
      assert(modules == Seq("DataTypeParameterized_Valid", "Top"))
    }

    it("should work for by name Data gen parameters") {
      val modules = convert(new Top {
        val inst0 = Instantiate(new DataTypeParameterizedByName(UInt(8.W)))
        val inst1 = Instantiate(new DataTypeParameterizedByName(UInt(8.W)))
      }).modules.map(_.name)
      assert(modules == Seq("DataTypeParameterizedByName_UInt8", "Top"))
    }
  }

  describe("The Instantiate cache") {
    it("should NOT be shared between elaborations within the same JVM run") {
      class MyTop extends Top {
        val inst = Instantiate(new OneArg(3))
      }
      val modules = convert(new MyTop).modules.map(_.name)
      assert(modules == Seq("OneArg", "Top"))
      // Building the same thing a second time should work
      val modules2 = convert(new MyTop).modules.map(_.name)
      assert(modules2 == Seq("OneArg", "Top"))
    }

    it("should properly handle case objects as parameters") {
      class MyTop extends Top {
        val inst0 = Instantiate(new ModuleParameterizedByProductTypes(FooEnum))
        val inst1 = Instantiate(new ModuleParameterizedByProductTypes(BarEnum))
      }
      val modules = convert(new MyTop).modules.map(_.name)
      assert(
        modules == Seq("ModuleParameterizedByProductTypes_FooEnum", "ModuleParameterizedByProductTypes_BarEnum", "Top")
      )
    }

    it("should properly handle case classes as parameters") {
      class MyTop extends Top {
        val inst0 = Instantiate(new ModuleParameterizedByProductTypes(FizzEnum(3)))
        val inst1 = Instantiate(new ModuleParameterizedByProductTypes(BuzzEnum(3)))
      }
      val modules = convert(new MyTop).modules.map(_.name)
      assert(
        modules == Seq(
          "ModuleParameterizedByProductTypes_FizzEnum3",
          "ModuleParameterizedByProductTypes_BuzzEnum3",
          "Top"
        )
      )
    }

    it("should properly handle Iterables") {
      class MyTop extends Top {
        val inst0 = Instantiate(new ModuleParameterizedBySeq(List(1, 2, 3)))
        val inst1 = Instantiate(new ModuleParameterizedBySeq(Vector(1, 2, 3)))
      }
      val modules = convert(new MyTop).modules.map(_.name)
      assert(
        modules == Seq(
          "ModuleParameterizedBySeq_1_2_3",
          "Top"
        )
      )
    }
  }

  describe("Instantiate") {
    it("should provide source locators for module instances") {
      // Materialize the source info so we can use it in the check
      implicit val info = implicitly[chisel3.experimental.SourceInfo]
      val chirrtl = convert(new Top {
        val inst = Instantiate(new OneArg(3))
      }).serialize
      chirrtl should include(s"inst inst of OneArg ${info.makeMessage(x => x)}")
    }

    it("should support BlackBoxes") {
      val modules = convert(new Top {
        val inst0 = Instantiate(new InstantiableBlackBox)
        val inst1 = Instantiate(new InstantiableBlackBox)
      }).modules.map(_.name)
      assert(modules == Seq("InstantiableBlackBox", "Top"))
    }

    it("should support ExtModules") {
      val modules = convert(new Top {
        val inst0 = Instantiate(new InstantiableExtModule)
        val inst1 = Instantiate(new InstantiableExtModule)
      }).modules.map(_.name)
      assert(modules == Seq("InstantiableExtModule", "Top"))
    }

    it("should support Intrinsics") {
      val modules = convert(new Top {
        val inst0 = Instantiate(new InstantiableIntrinsic)
        val inst1 = Instantiate(new InstantiableIntrinsic)
      }).modules.map(_.name)
      assert(modules == Seq("InstantiableIntrinsic", "Top"))
    }
  }

  describe("Arguments not of the proper form `new ModuleSubclass(...)(...)`") {
    it("should NOT compile if the `new Module` call is outside `Instantiate(...)`") {
      """val modules = convert(new Top {
        val gen0 = new NoArgs
        val gen1 = new NoArgs
        val inst0 = Instantiate(gen0)
        val inst1 = Instantiate(gen1)
      }).modules.map(_.name)
      assert(modules == Seq("Should", "Not", "Get", "Here"))
      """ shouldNot compile
    }
    it("should NOT compile if what we are Instantiating is not a Module") {
      """val modules = convert(new Top {
        class NotAModule(n: Int){
          val foo = n
        }
        val inst0 = Instantiate(new NotAModule(3))
        val inst1 = Instantiate(new NotAModule(3))
      }).modules.map(_.name)
      assert(modules == Seq("Should", "Not", "Get", "Here"))
      """ shouldNot compile
    }
  }

  it("Should make different Modules with reset type as a parameter") {
    class MyTop extends Top {
      withReset(reset.asAsyncReset) {
        val inst0 = Instantiate(new ParameterizedReset(true))
        val inst1 = Instantiate(new ParameterizedReset(true))
      }
      val inst2 = Instantiate(new ParameterizedReset(false))
      val inst3 = Instantiate(new ParameterizedReset(false))

      a[ChiselException] should be thrownBy {
        val inst4 = Instantiate(new ParameterizedReset(true))
      }
      a[ChiselException] should be thrownBy {
        withReset(reset.asAsyncReset) {
          val inst5 = Instantiate(new ParameterizedReset(false))
        }
      }
    }
    val modules = convert(new MyTop).modules.map(_.name)
    assert(
      modules == Seq(
        "ParameterizedReset",
        "ParameterizedReset_1",
        "Top"
      )
    )
  }
}

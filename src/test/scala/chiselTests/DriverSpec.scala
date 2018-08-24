// See LICENSE for license details.

package chiselTests

import java.io.File

import chisel3._
import chisel3.experimental.RawModule
import firrtl.{FirrtlExecutionSuccess, HasFirrtlExecutionOptions, FIRRTLException, TopNameAnnotation,
  TargetDirAnnotation, CompilerNameAnnotation, AnnotationSeq}
import firrtl.annotations.Annotation
import firrtl.options.ExecutionOptionsManager
import org.scalatest.{FreeSpec, Matchers, Succeeded}

class DummyModule extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(1.W))
    val out = Output(UInt(1.W))
  })
  io.out := io.in
}

class DriverSpec extends FreeSpec with Matchers with BackendCompilationUtilities {

  val name = "DummyModule"

  /** Check that a sequence of files DOES exist
    * @param exts files that should exist
    * @param dir an implicit directory name to search in
    * @return nothing as this is constructing a ScalaTest test
    */
  def filesShouldExist(exts: Seq[String])(implicit dir: String): Unit = exts
    .foreach{ ext =>
      val dummyOutput = new File(dir, name + "." + ext)
      info(s"DOES exist: $dummyOutput")
      dummyOutput.exists() should be (true)
      dummyOutput.delete() }

  /** Check that a sequence of files DOES NOT exist
    * @param exts files that shouldn't exist
    * @param dir an implicit directory name to search in
    * @return nothing as this is constructing a ScalaTest test
    */
  def filesShouldNotExist(exts: Seq[String])(implicit dir: String): Unit = exts
    .foreach{ ext =>
      val dummyOutput = new File(dir, name + "." + ext)
      info(s"DOES NOT exist: $dummyOutput")
      dummyOutput.exists should be (false) }

  /** Break up a map of equivalent options/annotations into only options or annotations
    * @param a a map of options to annotations
    * @note It may make sense to change this to compute the cross product of all options
    */
  def optionsOrAnnotations(a: Map[Array[String], Annotation]): Seq[(Array[String], AnnotationSeq)] = Seq(
    (a.keys.toArray.flatten, Seq.empty),
    (Array.empty, a.values.toSeq)
  )

  /** Pretty prints options/arguments as only their options
    * @param args an array of arguments
    */
  def collectOptions(args: Array[String]): String = args.filter(_.startsWith("-")).mkString(", ")

  /** Pretty prints annotations as their class names
    * @param annos a sequeunce of annotations
    */
  def collectAnnotations(annos: AnnotationSeq): String = annos.map(_.getClass.getSimpleName).mkString(", ")

  "Driver's execute methods are used to run chisel and firrtl" - {
    "options can be picked up from the command line with no args" in {
      // NOTE: Since we don't provide any arguments (notably, "--target-dir"),
      //  the generated files will be created in the current directory.
      implicit val targetDir = "."
      Driver.execute(Array.empty[String], () => new DummyModule) match {
        case ChiselExecutionSuccess(_, _, Some(_: FirrtlExecutionSuccess)) =>
          filesShouldExist(List("anno.json", "fir", "v"))
          succeed
        case f =>
          fail
      }
    }

    "options can be picked up from the command line setting top name" - {
      implicit val targetDir = createTestDirectory(this.getClass.getSimpleName).toString
      val optionsAnnos = Map(Array("-tn", name)         -> TopNameAnnotation(name),
                             Array("-td", targetDir)    -> TargetDirAnnotation(targetDir),
                             Array("--compiler", "low") -> CompilerNameAnnotation("low"))
      optionsOrAnnotations(optionsAnnos).map{ case (args, annos) =>
        s"""For options: "${collectOptions(args)}", annotations: "${collectAnnotations(annos)}"""" in {
          Driver.execute(args, () => new DummyModule, annos) match {
            case ChiselExecutionSuccess(_, _, Some(_: FirrtlExecutionSuccess)) =>
              filesShouldExist(List("anno.json", "fir", "lo.fir"))
              filesShouldNotExist(List("v", "hi.fir", "mid.fir"))
              succeed
            case x =>
              fail
          }
        }
      }
    }

    "option/arg --dont-save-chirrtl should disable CHIRRTL emission" - {
      implicit val targetDir = createTestDirectory(this.getClass.getSimpleName).toString
      val optionsAnnos = Map(Array("--dont-save-chirrtl")     -> DontSaveChirrtlAnnotation,
                             Array("--compiler", "middle")    -> CompilerNameAnnotation("middle"),
                             Array("--target-dir", targetDir) -> TargetDirAnnotation(targetDir))
      optionsOrAnnotations(optionsAnnos).map{ case (args, annos) =>
        s"""For options: "${collectOptions(args)}", annotations: "${collectAnnotations(annos)}"""" in {
          Driver.execute(args, () => new DummyModule, annos) match {
            case ChiselExecutionSuccess(_, _, Some(_: FirrtlExecutionSuccess)) =>
              filesShouldExist(List("anno.json", "mid.fir"))
              filesShouldNotExist(List("fir", "v", "hi.fir", "lo.fir"))
              succeed
            case _ =>
              fail
          }
        }
      }
    }

    "option/arg --dont-save-annotations should disable annotation emission" - {
      implicit val targetDir = createTestDirectory(this.getClass.getSimpleName).toString
      val optionsAnnos = Map(Array("--dont-save-annotations") -> DontSaveAnnotationsAnnotation,
                             Array("--compiler", "high")      -> CompilerNameAnnotation("high"),
                             Array("--target-dir", targetDir) -> TargetDirAnnotation(targetDir))
      optionsOrAnnotations(optionsAnnos).map{ case (args, annos) =>
        s"""For options: "${collectOptions(args)}", annotations: "${collectAnnotations(annos)}"""" in {
          Driver.execute(args, () => new DummyModule, annos) match {
            case ChiselExecutionSuccess(_, _, Some(_: FirrtlExecutionSuccess)) =>
              filesShouldExist(List("hi.fir"))
              filesShouldNotExist(List("v", "lo.fir", "mid.fir", "anno.json"))
              succeed
            case _ =>
              fail
          }
        }
      }
    }

    "option/arg --no-run-firrtl should emit CHIRRTL and not FIRRTL or Verilog" - {
      implicit val targetDir = createTestDirectory(this.getClass.getSimpleName).toString
      val optionsAnnos = Map(Array("--no-run-firrtl")         -> NoRunFirrtlAnnotation,
                             Array("--compiler", "verilog")   -> CompilerNameAnnotation("verilog"),
                             Array("--target-dir", targetDir) -> TargetDirAnnotation(targetDir))
      optionsOrAnnotations(optionsAnnos).foreach{ case (args, annos) =>
        s"""For options: "${collectOptions(args)}", annotations: "${collectAnnotations(annos)}"""" in {
          Driver.execute(args, () => new DummyModule, annos) match {
            case ChiselExecutionSuccess(_, _, None) =>
              filesShouldExist(List("anno.json", "fir"))
              filesShouldNotExist(List("v", "hi.fir", "lo.fir", "mid.fir"))
              succeed
            case _ =>
              fail
          }
        }
      }
    }

    "execute returns a chisel execution result" in {
      val targetDir = createTestDirectory(this.getClass.getSimpleName).toString
      val args = Array("--dut", "chiselTests.DummyModule", "--compiler", "low", "--target-dir", targetDir)
      val result = Driver.execute(args, Seq[Annotation]())
      result shouldBe a[ChiselExecutionSuccess]
      val successResult = result.asInstanceOf[ChiselExecutionSuccess]
      successResult.emitted should include ("circuit DummyModule")
      val dummyOutput = new File(targetDir, "DummyModule.lo.fir")
      dummyOutput.exists() should be(true)
      dummyOutput.delete()
    }
  }

  "Invalid options should be caught when" - {
    def shouldExceptOnOptionsOrAnnotations(name: String, args: Array[String], annos: Seq[Annotation]) {
      name in {
        info("via annotations")
        a [ChiselOptionsException] should be thrownBy (Driver.execute(Array[String](), Seq.fill(2)(annos).flatten))
        info("via arguments")
        a [ChiselOptionsException] should be thrownBy (Driver.execute(Array.fill(2)(args).flatten, Seq[Annotation]()))
        info("via arguments and annotations")
        a [ChiselOptionsException] should be thrownBy (Driver.execute(args, annos))
      }
    }

    "no Chisel circuit is specified" in {
      a [ChiselException] should be thrownBy (Driver.execute(Array[String](), Seq[Annotation]()))
    }

    shouldExceptOnOptionsOrAnnotations("multiple Chisel circuits are specified",
                                       Array("--dut", "chiselTests.DummyModule"),
                                       Seq(ChiselCircuitAnnotation(() => new DummyModule)))
  }
}

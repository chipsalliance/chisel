// SPDX-License-Identifier: Apache-2.0

package chiselTests.simulator

import java.io.File
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers
import chisel3.RawModule
import chisel3.layer.{addLayer, Layer, LayerConfig}
import chisel3.simulator.{ChiselWorkspace, LayerControl}
import svsim.Workspace

class LayerControlSpec extends AnyFunSpec with Matchers {

  object A extends Layer(LayerConfig.Extract())
  object B extends Layer(LayerConfig.Extract()) {
    object C extends Layer(LayerConfig.Extract())
  }
  class Foo extends RawModule {
    Seq(A, B, B.C).foreach(addLayer)
  }

  val workspace = new Workspace(path = "test_run_dir/LayerControlSpec")
  workspace.reset()
  val elaboratedModule = workspace.elaborateGeneratedModule({ () => new Foo })

  describe("LayerControl.EnableAll") {
    it("should include all layer files") {
      val layerControl = LayerControl.EnableAll

      info("non-layer files are ignored")
      layerControl.shouldIncludeFile(elaboratedModule).isDefinedAt(new File("Foo.sv")) should be(false)
      layerControl.shouldIncludeDirectory(elaboratedModule, "build").isDefinedAt(new File("build/foo")) should be(false)

      info("layer ABI files are included")
      Seq("layers-Foo-A.sv", "layers-Foo-B.sv", "layers-Foo-B-C.sv").map(new File(_)).foreach { case filename =>
        info(s"$filename is included")
        layerControl.shouldIncludeFile(elaboratedModule)(filename) should be(true)
      }
      info("layer directories are included")
      Seq("build/A", "build/B", "build/B/C").map(new File(_)).foreach { case directory =>
        info(s"$directory is included")
        layerControl.shouldIncludeDirectory(elaboratedModule, "build")(directory) should be(true)
      }
    }
  }
  describe("LayerControl.Enable()") {
    it("should include no layer files") {
      val layerControl = LayerControl.Enable()

      info("non-layer files are ignored")
      layerControl.shouldIncludeFile(elaboratedModule).isDefinedAt(new File("Foo.sv")) should be(false)
      layerControl.shouldIncludeDirectory(elaboratedModule, "build").isDefinedAt(new File("build/foo")) should be(false)

      info("layer ABI files are excluded")
      Seq("layers-Foo-A.sv", "layers-Foo-B.sv", "layers-Foo-B-C.sv").map(new File(_)).foreach { case filename =>
        info(s"$filename is excluded")
        layerControl.shouldIncludeFile(elaboratedModule)(filename) should be(false)
      }
      info("layer directories are excluded")
      Seq("build/A", "build/B", "build/B/C").map(new File(_)).foreach { case directory =>
        info(s"$directory is excluded")
        layerControl.shouldIncludeDirectory(elaboratedModule, "build")(directory) should be(false)
      }
    }
  }
  describe("LayerControl.DisableAll") {
    it("should include no layer files") {
      val layerControl = LayerControl.Enable()

      info("non-layer files are ignored")
      layerControl.shouldIncludeFile(elaboratedModule).isDefinedAt(new File("Foo.sv")) should be(false)
      layerControl.shouldIncludeDirectory(elaboratedModule, "build").isDefinedAt(new File("build/foo")) should be(false)

      info("layer ABI files are excluded")
      Seq("layers-Foo-A.sv", "layers-Foo-B.sv", "layers-Foo-B-C.sv").map(new File(_)).foreach { case filename =>
        info(s"$filename is excluded")
        layerControl.shouldIncludeFile(elaboratedModule)(filename) should be(false)
      }
      info("layer directories are excluded")
      Seq("build/A", "build/B", "build/B/C").map(new File(_)).foreach { case directory =>
        info(s"$directory is excluded")
        layerControl.shouldIncludeDirectory(elaboratedModule, "build")(directory) should be(false)
      }
    }
  }
  describe("LayerControl.Enable(A, B.C)") {
    it("should include only specified layers") {
      val layerControl = LayerControl.Enable(A, B.C)

      info("non-layer files are ignored")
      layerControl.shouldIncludeFile(elaboratedModule).isDefinedAt(new File("foo")) should be(false)
      layerControl.shouldIncludeDirectory(elaboratedModule, "build").isDefinedAt(new File("build/foo")) should be(false)

      info("layer ABI files are excluded or excluded appropriately")
      Seq("layers-Foo-A.sv", "layers-Foo-B-C.sv").map(new File(_)).foreach { case filename =>
        info(s"$filename is included")
        layerControl.shouldIncludeFile(elaboratedModule)(filename) should be(true)
      }

      info("layers-Foo-A-B.sv is excluded")
      layerControl.shouldIncludeFile(elaboratedModule)(new File("layers-Foo-A-B.sv")) should be(false)

      info("layer directories are excluded or excluded appropriately")
      Seq("build/A", "build/B/C").map(new File(_)).foreach { case directory =>
        info(s"$directory is included")
        layerControl.shouldIncludeDirectory(elaboratedModule, "build")(directory) should be(true)
      }

      info("build/B is excluded")
      layerControl.shouldIncludeDirectory(elaboratedModule, "build")(new File("build/B")) should be(false)
    }
  }
  describe("LayerControl.Disable()") {
    it("should include all layer files") {
      val layerControl = LayerControl.Disable()

      info("non-layer files are ignored")
      layerControl.shouldIncludeFile(elaboratedModule).isDefinedAt(new File("Foo.sv")) should be(false)
      layerControl.shouldIncludeDirectory(elaboratedModule, "build").isDefinedAt(new File("build/foo")) should be(false)

      info("layer ABI files are included")
      Seq("layers-Foo-A.sv", "layers-Foo-B.sv", "layers-Foo-B-C.sv").map(new File(_)).foreach { case filename =>
        info(s"$filename is included")
        layerControl.shouldIncludeFile(elaboratedModule)(filename) should be(true)
      }
      info("layer directories are included")
      Seq("build/A", "build/B", "build/B/C").map(new File(_)).foreach { case directory =>
        info(s"$directory is included")
        layerControl.shouldIncludeDirectory(elaboratedModule, "build")(directory) should be(true)
      }
    }
  }
  describe("LayerControl.Disable(A, B.C)") {
    it("should include only specified layers") {
      val layerControl = LayerControl.Disable(A, B.C)

      info("non-layer files are ignored")
      layerControl.shouldIncludeFile(elaboratedModule).isDefinedAt(new File("foo")) should be(false)
      layerControl.shouldIncludeDirectory(elaboratedModule, "build").isDefinedAt(new File("build/foo")) should be(false)

      info("layer ABI files are excluded or excluded appropriately")
      layerControl.shouldIncludeFile(elaboratedModule)(new File("layers-Foo-B.sv")) should be(true)
      info("layers-Foo-A-B.sv is excluded")
      Seq("layers-Foo-A.sv", "layers-Foo-B-C.sv").map(new File(_)).foreach { case filename =>
        info(s"$filename is excluded")
        layerControl.shouldIncludeFile(elaboratedModule)(filename) should be(false)
      }

      info("layer directories are excluded or excluded appropriately")
      info("build/B is included")
      layerControl.shouldIncludeDirectory(elaboratedModule, "build")(new File("build/B")) should be(true)
      Seq("build/A", "build/B/C").map(new File(_)).foreach { case directory =>
        info(s"$directory is excluded")
        layerControl.shouldIncludeDirectory(elaboratedModule, "build")(directory) should be(false)
      }
    }
  }
}

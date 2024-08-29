// SPDX-License-Identifier: Apache-2.0

package chiselTests.simulator

import java.io.File
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers
import chisel3.layer.{Layer, LayerConfig}
import chisel3.simulator.LayerControl

class LayerControlSpec extends AnyFunSpec with Matchers {
  describe("LayerControl.EnableAll") {
    it("should always filter to true") {
      val layerControl = LayerControl.EnableAll
      layerControl.filter(new File("foo")) should be(true)
      layerControl.filter(new File("layers-foo-bar.sv")) should be(true)
    }
  }
  describe("LayerControl.Enable()") {
    it("should return true for non-layers and false for layers") {
      val layerControl = LayerControl.Enable()
      layerControl.filter(new File("foo")) should be(true)
      layerControl.filter(new File("layers-foo-bar.sv")) should be(false)
    }
  }
  describe("LayerControl.DisableAll") {
    it("should return true for non-layers and false for layers") {
      LayerControl.DisableAll.filter(new File("foo")) should be(true)
      LayerControl.DisableAll.filter(new File("layers-foo-bar.sv")) should be(false)
    }
  }
  describe("LayerControl.Enable") {
    it("should return true for non-layers and filter layers properly") {
      object A extends Layer(LayerConfig.Extract())
      object B extends Layer(LayerConfig.Extract()) {
        object C extends Layer(LayerConfig.Extract())
      }
      val layerControl = LayerControl.Enable(A, B.C)
      layerControl.filter(new File("foo")) should be(true)
      layerControl.filter(new File("layers-foo.sv")) should be(false)
      layerControl.filter(new File("layers-foo-A.sv")) should be(true)
      layerControl.filter(new File("layers-foo-A-B.sv")) should be(false)
      layerControl.filter(new File("layers-foo-B-C.sv")) should be(true)
    }
  }
}

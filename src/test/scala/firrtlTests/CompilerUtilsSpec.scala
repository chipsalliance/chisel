// See LICENSE for license details.

package firrtlTests

import firrtl._
import firrtl.CompilerUtils.mergeTransforms

class CompilerUtilsSpec extends FirrtlFlatSpec {

  def genTransform(_inputForm: CircuitForm, _outputForm: CircuitForm) = new Transform {
    def inputForm = _inputForm
    def outputForm = _outputForm
    def execute(state: CircuitState): CircuitState = state
  }

  // Core lowering transforms
  val chirrtlToHigh = genTransform(ChirrtlForm, HighForm)
  val highToMid = genTransform(HighForm, MidForm)
  val midToLow = genTransform(MidForm, LowForm)
  val chirrtlToLowList = List(chirrtlToHigh, highToMid, midToLow)

  // Custom transforms
  val chirrtlToChirrtl = genTransform(ChirrtlForm, ChirrtlForm)
  val highToHigh = genTransform(HighForm, HighForm)
  val midToMid = genTransform(MidForm, MidForm)
  val lowToLow = genTransform(LowForm, LowForm)

  val lowToHigh = genTransform(LowForm, HighForm)

  val lowToLowTwo = genTransform(LowForm, LowForm)

  behavior of "mergeTransforms"

  it should "do nothing if there are no custom transforms" in {
    mergeTransforms(chirrtlToLowList, List.empty) should be (chirrtlToLowList)
  }

  it should "insert transforms at the correct place" in {
    mergeTransforms(chirrtlToLowList, List(chirrtlToChirrtl)) should be
      (chirrtlToChirrtl +: chirrtlToLowList)
    mergeTransforms(chirrtlToLowList, List(highToHigh)) should be
      (List(chirrtlToHigh, highToHigh, highToMid, midToLow))
    mergeTransforms(chirrtlToLowList, List(midToMid)) should be
      (List(chirrtlToHigh, highToMid, midToMid, midToLow))
    mergeTransforms(chirrtlToLowList, List(lowToLow)) should be
      (chirrtlToLowList :+ lowToLow)
  }

  it should "insert transforms at the last legal location" in {
    lowToLow should not be (lowToLowTwo) // sanity check
    mergeTransforms(chirrtlToLowList :+ lowToLow, List(lowToLowTwo)).last should be (lowToLowTwo)
  }

  it should "insert multiple transforms correctly" in {
    mergeTransforms(chirrtlToLowList, List(highToHigh, lowToLow)) should be
      (List(chirrtlToHigh, highToHigh, highToMid, midToLow, lowToLow))
  }

  it should "handle transforms that raise the form" in {
    mergeTransforms(chirrtlToLowList, List(lowToHigh)) match {
      case chirrtlToHigh :: highToMid :: midToLow :: lowToHigh :: remainder =>
        // Remainder will be the actual Firrtl lowering transforms
        remainder.head.inputForm should be (HighForm)
        remainder.last.outputForm should be (LowForm)
      case _ => fail()
    }
  }

  // Order is not always maintained, see note on function Scaladoc
  it should "maintain order of custom tranforms" in {
    mergeTransforms(chirrtlToLowList, List(lowToLow, lowToLowTwo)) should be
      (chirrtlToLowList ++ List(lowToLow, lowToLowTwo))
  }

}


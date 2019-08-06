--------------------------
src/test/scala/chiselTests
--------------------------

.. toctree::
	experimental/experimental.rst
	util/util.rst
	stage/stage.rst


Decoder.scala
-------------
.. chisel:attr:: class Decoder(bitpats: List[String]) extends Module


.. chisel:attr:: class DecoderTester(pairs: List[(String, String)]) extends BasicTester


.. chisel:attr:: class DecoderSpec extends ChiselPropSpec


ComplexAssign.scala
-------------------
.. chisel:attr:: class Complex[T <: Data](val re: T, val im: T) extends Bundle


.. chisel:attr:: class ComplexAssign(w: Int) extends Module


.. chisel:attr:: class ComplexAssignTester(enList: List[Boolean], re: Int, im: Int) extends BasicTester


.. chisel:attr:: class ComplexAssignSpec extends ChiselPropSpec


BitwiseOps.scala
----------------
.. chisel:attr:: class BitwiseOpsTester(w: Int, _a: Int, _b: Int) extends BasicTester


.. chisel:attr:: class BitwiseOpsSpec extends ChiselPropSpec


When.scala
----------
.. chisel:attr:: class WhenTester() extends BasicTester


.. chisel:attr:: class OverlappedWhenTester() extends BasicTester


.. chisel:attr:: class NoOtherwiseOverlappedWhenTester() extends BasicTester


.. chisel:attr:: class SubmoduleWhenTester extends BasicTester


.. chisel:attr:: class WhenSpec extends ChiselFlatSpec


ModuleExplicitResetSpec.scala
-----------------------------
.. chisel:attr:: class ModuleExplicitResetSpec extends ChiselFlatSpec


MulLookup.scala
---------------
.. chisel:attr:: class MulLookup(val w: Int) extends Module


.. chisel:attr:: class MulLookupTester(w: Int, x: Int, y: Int) extends BasicTester


.. chisel:attr:: class MulLookupSpec extends ChiselPropSpec


Assert.scala
------------
.. chisel:attr:: class FailingAssertTester() extends BasicTester


.. chisel:attr:: class SucceedingAssertTester() extends BasicTester


.. chisel:attr:: class PipelinedResetModule extends Module


.. chisel:attr:: class PipelinedResetTester extends BasicTester


.. chisel:attr:: class ModuloAssertTester extends BasicTester


.. chisel:attr:: class FormattedAssertTester extends BasicTester


.. chisel:attr:: class BadUnescapedPercentAssertTester extends BasicTester


.. chisel:attr:: class AssertSpec extends ChiselFlatSpec


BlackBoxImpl.scala
------------------
.. chisel:attr:: class BlackBoxAdd(n : Int) extends HasBlackBoxInline


.. chisel:attr:: class UsesBlackBoxAddViaInline extends Module


.. chisel:attr:: class BlackBoxMinus extends HasBlackBoxResource


.. chisel:attr:: class BlackBoxMinusPath extends HasBlackBoxPath


.. chisel:attr:: class UsesBlackBoxMinusViaResource extends Module


.. chisel:attr:: class UsesBlackBoxMinusViaPath extends Module


.. chisel:attr:: class BlackBoxImplSpec extends FreeSpec with Matchers


AnnotatingDiamondSpec.scala
---------------------------
.. chisel:attr:: case class IdentityAnnotation(target: Named, value: String) extends SingleTargetAnnotation[Named]

	These annotations and the IdentityTransform class serve as an example of how to write a	Chisel/Firrtl library
  

.. chisel:attr:: case class IdentityChiselAnnotation(target: InstanceId, value: String) extends ChiselAnnotation with RunFirrtlTransform

	ChiselAnnotation that corresponds to the above FIRRTL annotation 


.. chisel:attr:: object identify


.. chisel:attr:: class IdentityTransform extends Transform


.. chisel:attr:: class ModC(widthC: Int) extends Module

		This class has parameterizable widths, it will generate different hardware
	
	:param widthC: io width
	  

.. chisel:attr:: class ModA(annoParam: Int) extends Module

		instantiates a C of a particular size, ModA does not generate different hardware
	based on it's parameter
	
	:param annoParam:  parameter is only used in annotation not in circuit
	  

.. chisel:attr:: class ModB(widthB: Int) extends Module


.. chisel:attr:: class TopOfDiamond extends Module


.. chisel:attr:: class DiamondTester extends BasicTester


.. chisel:attr:: class AnnotatingDiamondSpec extends FreeSpec with Matchers


Reg.scala
---------
.. chisel:attr:: class RegSpec extends ChiselFlatSpec


.. chisel:attr:: class ShiftTester(n: Int) extends BasicTester


.. chisel:attr:: class ShiftResetTester(n: Int) extends BasicTester


.. chisel:attr:: class ShiftRegisterSpec extends ChiselPropSpec


UIntOps.scala
-------------
.. chisel:attr:: class UIntOps extends Module


.. chisel:attr:: class UIntOpsTester(a: Long, b: Long) extends BasicTester


.. chisel:attr:: class GoodBoolConversion extends Module


.. chisel:attr:: class BadBoolConversion extends Module


.. chisel:attr:: class NegativeShift(t: => Bits) extends Module


.. chisel:attr:: class UIntLitExtractTester extends BasicTester


.. chisel:attr:: class UIntOpsSpec extends ChiselPropSpec with Matchers


ImplicitConversionsSpec.scala
-----------------------------
.. chisel:attr:: class ImplicitConversionsSpec extends ChiselFlatSpec


BundleWire.scala
----------------
.. chisel:attr:: class Coord extends Bundle


.. chisel:attr:: class BundleWire(n: Int) extends Module


.. chisel:attr:: class BundleToUnitTester extends BasicTester


.. chisel:attr:: class BundleWireTester(n: Int, x: Int, y: Int) extends BasicTester


.. chisel:attr:: class BundleWireSpec extends ChiselPropSpec


.. chisel:attr:: class BundleToUIntSpec extends ChiselPropSpec


WireSpec.scala
--------------
.. chisel:attr:: class WireSpec extends ChiselFlatSpec


ConnectSpec.scala
-----------------
.. chisel:attr:: abstract class CrossCheck extends Bundle


.. chisel:attr:: class CrossConnects(inType: Data, outType: Data) extends Module


.. chisel:attr:: class PipeInternalWires extends Module


.. chisel:attr:: class CrossConnectTester(inType: Data, outType: Data) extends BasicTester


.. chisel:attr:: class ConnectSpec extends ChiselPropSpec


AsTypeOfTester.scala
--------------------
.. chisel:attr:: class AsTypeOfBundleTester extends BasicTester


.. chisel:attr:: class AsTypeOfBundleZeroWidthTester extends BasicTester


.. chisel:attr:: class AsTypeOfVecTester extends BasicTester


.. chisel:attr:: class AsTypeOfTruncationTester extends BasicTester


.. chisel:attr:: class ResetAsTypeOfBoolTester extends BasicTester


.. chisel:attr:: class AsChiselEnumTester extends BasicTester


.. chisel:attr:: class AsTypeOfSpec extends ChiselFlatSpec


MissingCloneBindingExceptionSpec.scala
--------------------------------------
.. chisel:attr:: class MissingCloneBindingExceptionSpec extends ChiselFlatSpec with Matchers


InvalidateAPISpec.scala
-----------------------
.. chisel:attr:: class InvalidateAPISpec extends ChiselPropSpec with Matchers with BackendCompilationUtilities


DriverSpec.scala
----------------
.. chisel:attr:: class DummyModule extends Module


.. chisel:attr:: class DriverSpec extends FreeSpec with Matchers


BetterNamingTests.scala
-----------------------
.. chisel:attr:: class Other(w: Int) extends Module


.. chisel:attr:: class PerNameIndexing(count: Int) extends NamedModuleTester


.. chisel:attr:: class IterableNaming extends NamedModuleTester


.. chisel:attr:: class DigitFieldNamesInRecord extends NamedModuleTester


.. chisel:attr:: class BetterNamingTests extends ChiselFlatSpec


SwitchSpec.scala
----------------
.. chisel:attr:: class SwitchSpec extends ChiselFlatSpec


PrintableSpec.scala
-------------------
.. chisel:attr:: class PrintableSpec extends FlatSpec with Matchers


Vec.scala
---------
.. chisel:attr:: class LitTesterMod(vecSize: Int) extends Module


.. chisel:attr:: class RegTesterMod(vecSize: Int) extends Module


.. chisel:attr:: class IOTesterMod(vecSize: Int) extends Module


.. chisel:attr:: class OneBitUnitRegVec extends Module


.. chisel:attr:: class LitTester(w: Int, values: List[Int]) extends BasicTester


.. chisel:attr:: class RegTester(w: Int, values: List[Int]) extends BasicTester


.. chisel:attr:: class IOTester(w: Int, values: List[Int]) extends BasicTester


.. chisel:attr:: class IOTesterModFill(vecSize: Int) extends Module


.. chisel:attr:: class ValueTester(w: Int, values: List[Int]) extends BasicTester


.. chisel:attr:: class TabulateTester(n: Int) extends BasicTester


.. chisel:attr:: class ShiftRegisterTester(n: Int) extends BasicTester


.. chisel:attr:: class HugeVecTester(n: Int) extends BasicTester


.. chisel:attr:: class OneBitUnitRegVecTester extends BasicTester


.. chisel:attr:: class ZeroEntryVecTester extends BasicTester


.. chisel:attr:: class PassthroughModuleTester extends Module


.. chisel:attr:: class ModuleIODynamicIndexTester(n: Int) extends BasicTester


.. chisel:attr:: class VecSpec extends ChiselPropSpec


DataPrint.scala
---------------
.. chisel:attr:: class DataPrintSpec extends ChiselFlatSpec with Matchers


VectorPacketIO.scala
--------------------
.. chisel:attr:: class Packet extends Bundle

		This test used to fail when assignment statements were
	contained in DeqIO and EnqIO constructors.
	The symptom is creation of a firrtl file
	with missing declarations, the problem is exposed by
	the creation of the val outs in VectorPacketIO
	
	NOTE: The problem does not exist now because the initialization
	code has been removed from DeqIO and EnqIO
	
	IMPORTANT:  The canonical way to initialize a decoupled inteface is still being debated.
  

.. chisel:attr:: class VectorPacketIO(val n: Int) extends Bundle

		The problem occurs with just the ins or the outs
	lines also.
	The problem does not occur if the Vec is taken out
  

.. chisel:attr:: class BrokenVectorPacketModule extends Module

		a module uses the vector based IO bundle
	the value of n does not affect the error
  

.. chisel:attr:: class VectorPacketIOUnitTester extends BasicTester


.. chisel:attr:: class VectorPacketIOUnitTesterSpec extends ChiselFlatSpec


BundleSpec.scala
----------------
.. chisel:attr:: trait BundleSpecUtils


.. chisel:attr:: class BundleSpec extends ChiselFlatSpec with BundleSpecUtils


TransitNameSpec.scala
---------------------
.. chisel:attr:: class TransitNameSpec extends FlatSpec with Matchers


.. chisel:attr:: class Top extends RawModule

	A top-level module that instantiates three copies of MyModule 


WidthSpec.scala
---------------
.. chisel:attr:: class SimpleBundle extends Bundle


.. chisel:attr:: object SimpleBundle


.. chisel:attr:: class WidthSpec extends ChiselFlatSpec


.. chisel:attr:: abstract class WireRegWidthSpecImpl extends ChiselFlatSpec


.. chisel:attr:: class WireWidthSpec extends WireRegWidthSpecImpl


.. chisel:attr:: class RegWidthSpec extends WireRegWidthSpecImpl


.. chisel:attr:: abstract class WireDefaultRegInitSpecImpl extends ChiselFlatSpec


.. chisel:attr:: class WireDefaultWidthSpec extends WireDefaultRegInitSpecImpl


.. chisel:attr:: class RegInitWidthSpec extends WireDefaultRegInitSpecImpl


BundleLiteralSpec.scala
-----------------------
.. chisel:attr:: class BundleLiteralSpec extends ChiselFlatSpec


CompatibilitySpec.scala
-----------------------
.. chisel:attr:: class CompatibiltySpec extends ChiselFlatSpec with GeneratorDrivenPropertyChecks


ParameterizedModule.scala
-------------------------
.. chisel:attr:: class ParameterizedModule(invert: Boolean) extends Module


.. chisel:attr:: class ParameterizedModuleTester() extends BasicTester

	A simple test to check Module deduplication doesn't affect correctness (two	modules with the same name but different contents aren't aliased). Doesn't
	check that deduplication actually happens, though.
  

.. chisel:attr:: class ParameterizedModuleSpec extends ChiselFlatSpec


FixedPointSpec.scala
--------------------
.. chisel:attr:: class FixedPointLiteralSpec extends FlatSpec with Matchers


.. chisel:attr:: class FixedPointFromBitsTester extends BasicTester


.. chisel:attr:: class FixedPointMuxTester extends BasicTester


.. chisel:attr:: class SBP extends Module


.. chisel:attr:: class SBPTester extends BasicTester


.. chisel:attr:: class FixedPointLitExtractTester extends BasicTester


.. chisel:attr:: class FixedPointSpec extends ChiselPropSpec


Risc.scala
----------
.. chisel:attr:: class Risc extends Module


.. chisel:attr:: class RiscTester(c: Risc) extends Tester(c)


.. chisel:attr:: class RiscSpec extends ChiselPropSpec


AnalogIntegrationSpec.scala
---------------------------
.. chisel:attr:: class AnalogBlackBoxPort extends Bundle


.. chisel:attr:: class AnalogBlackBoxIO(val n: Int) extends Bundle


.. chisel:attr:: class AnalogBlackBox(index: Int) extends BlackBox(Map("index" -> index))


.. chisel:attr:: class AnalogBlackBoxModule(index: Int) extends Module


.. chisel:attr:: class AnalogBlackBoxWrapper(n: Int, idxs: Seq[Int]) extends Module


.. chisel:attr:: abstract class AnalogDUTModule(numBlackBoxes: Int) extends Module


.. chisel:attr:: class AnalogDUT extends AnalogDUTModule(5)

	Single test case for lots of things	
	$ - Wire at top connecting child inouts (Done in AnalogDUT)
	$ - Port inout connected to 1 or more children inouts (AnalogBackBoxWrapper)
	$ - Multiple port inouts connected (AnalogConnector)
  

.. chisel:attr:: class AnalogSmallDUT extends AnalogDUTModule(4)

	Same as :chisel:reref:`AnalogDUT`  except it omits :chisel:reref:`AnalogConnector`  because that is currently not	supported by Verilator
	@todo Delete once Verilator can handle :chisel:reref:`AnalogDUT`
  

.. chisel:attr:: class AnalogIntegrationTester(mod: => AnalogDUTModule) extends BasicTester


.. chisel:attr:: class AnalogIntegrationSpec extends ChiselFlatSpec


CompileOptionsTest.scala
------------------------
.. chisel:attr:: class CompileOptionsSpec extends ChiselFlatSpec


InlineSpec.scala
----------------
.. chisel:attr:: class InlineSpec extends FreeSpec with ChiselRunners with Matchers


MixedVecSpec.scala
------------------
.. chisel:attr:: class MixedVecAssignTester(w: Int, values: List[Int]) extends BasicTester


.. chisel:attr:: class MixedVecRegTester(w: Int, values: List[Int]) extends BasicTester


.. chisel:attr:: class MixedVecIOPassthroughModule[T <: Data](hvec: MixedVec[T]) extends Module


.. chisel:attr:: class MixedVecIOTester(boundVals: Seq[Data]) extends BasicTester


.. chisel:attr:: class MixedVecZeroEntryTester extends BasicTester


.. chisel:attr:: class MixedVecUIntDynamicIndexTester extends BasicTester


.. chisel:attr:: class MixedVecTestBundle extends Bundle


.. chisel:attr:: class MixedVecSmallTestBundle extends Bundle


.. chisel:attr:: class MixedVecFromVecTester extends BasicTester


.. chisel:attr:: class MixedVecConnectWithVecTester extends BasicTester


.. chisel:attr:: class MixedVecConnectWithSeqTester extends BasicTester


.. chisel:attr:: class MixedVecOneBitTester extends BasicTester


.. chisel:attr:: class MixedVecSpec extends ChiselPropSpec


RecordSpec.scala
----------------
.. chisel:attr:: final class CustomBundle(elts: (String, Data)*) extends Record


.. chisel:attr:: trait RecordSpecUtils


.. chisel:attr:: class RecordSpec extends ChiselFlatSpec with RecordSpecUtils


ChiselSpec.scala
----------------
.. chisel:attr:: trait ChiselRunners extends Assertions with BackendCompilationUtilities

	Common utility functions for Chisel unit tests. 


	.. chisel:attr:: def generateFirrtl(t: => RawModule): String

	
		Given a generator, return the Firrtl that it generates.	
		
		:param t: Module generator
		:return: Firrtl representation as a String
		    


	.. chisel:attr:: def compile(t: => RawModule): String =

	
		Compiles a Chisel Module to Verilog	NOTE: This uses the "test_run_dir" as the default directory for generated code.
		
		:param t: the generator for the module
		:return: the Verilog code as a string.
		    


.. chisel:attr:: abstract class ChiselFlatSpec extends FlatSpec with ChiselRunners with Matchers

	Spec base class for BDD-style testers. 


.. chisel:attr:: abstract class ChiselFlatSpec extends FlatSpec with ChiselRunners with Matchers  class ChiselTestUtilitiesSpec extends ChiselFlatSpec


.. chisel:attr:: class ChiselTestUtilitiesSpec extends ChiselFlatSpec


.. chisel:attr:: class ChiselPropSpec extends PropSpec with ChiselRunners with PropertyChecks with Matchers

	Spec base class for property-based testers. 


CloneModuleSpec.scala
---------------------
.. chisel:attr:: class MultiIOQueue[T <: Data](gen: T, val entries: Int) extends MultiIOModule


.. chisel:attr:: class QueueClone(multiIO: Boolean


.. chisel:attr:: class QueueCloneTester(x: Int, multiIO: Boolean


.. chisel:attr:: class CloneModuleSpec extends ChiselPropSpec


MultiClockSpec.scala
--------------------
.. chisel:attr:: class ClockDividerTest extends BasicTester

	Multi-clock test of a Reg using a different clock via withClock 


.. chisel:attr:: class MultiClockSubModuleTest extends BasicTester


.. chisel:attr:: class WithResetTest extends BasicTester

	Test withReset changing the reset of a Reg 


.. chisel:attr:: class MultiClockMemTest extends BasicTester

	Test Mem ports with different clocks 


.. chisel:attr:: class MultiClockSpec extends ChiselFlatSpec


CompatibilityInteroperabilitySpec.scala
---------------------------------------
.. chisel:attr:: object CompatibilityComponents


.. chisel:attr:: object Chisel3Components


.. chisel:attr:: class CompatibiltyInteroperabilitySpec extends ChiselFlatSpec


DedupSpec.scala
---------------
.. chisel:attr:: class DedupIO extends Bundle


.. chisel:attr:: class DedupQueues(n: Int) extends Module


.. chisel:attr:: class DedupSubModule extends Module


.. chisel:attr:: class NestedDedup extends Module


.. chisel:attr:: object DedupConsts


.. chisel:attr:: class SharedConstantValDedup extends Module


.. chisel:attr:: class SharedConstantValDedupTop extends Module


.. chisel:attr:: class DedupSpec extends ChiselFlatSpec


MultiAssign.scala
-----------------
.. chisel:attr:: class LastAssignTester() extends BasicTester


.. chisel:attr:: class MultiAssignSpec extends ChiselFlatSpec


.. chisel:attr:: class IllegalAssignSpec extends ChiselFlatSpec


Direction.scala
---------------
.. chisel:attr:: class DirectionedBundle extends Bundle


.. chisel:attr:: class DirectionHaver extends Module


.. chisel:attr:: class GoodDirection extends DirectionHaver


.. chisel:attr:: class BadDirection extends DirectionHaver


.. chisel:attr:: class BadSubDirection extends DirectionHaver


.. chisel:attr:: class TopDirectionOutput extends Module


.. chisel:attr:: class DirectionSpec extends ChiselPropSpec with Matchers


MultiIOModule.scala
-------------------
.. chisel:attr:: class MultiIOPlusOne extends MultiIOModule


.. chisel:attr:: class MultiIOTester extends BasicTester


.. chisel:attr:: trait LiteralOutputTrait extends MultiIOModule


.. chisel:attr:: trait MultiIOTrait extends MultiIOModule


.. chisel:attr:: class ComposedMultiIOModule extends MultiIOModule with LiteralOutputTrait with MultiIOTrait


.. chisel:attr:: class ComposedMultiIOTester extends BasicTester


.. chisel:attr:: class MultiIOSpec extends ChiselFlatSpec


BoringUtilsSpec.scala
---------------------
.. chisel:attr:: abstract class ShouldntAssertTester(cyclesToWait: BigInt


.. chisel:attr:: class StripNoDedupAnnotation extends Transform


.. chisel:attr:: class BoringUtilsSpec extends ChiselFlatSpec with ChiselRunners


.. chisel:attr:: class TopTester extends ShouldntAssertTester

	This is testing a complicated wiring pattern and exercising	the necessity of disabling deduplication for sources and sinks.
	Without disabling deduplication, this test will fail.
    

LiteralExtractorSpec.scala
--------------------------
.. chisel:attr:: class LiteralExtractorSpec extends ChiselFlatSpec


Stack.scala
-----------
.. chisel:attr:: class ChiselStack(val depth: Int) extends Module


.. chisel:attr:: class StackTester(c: Stack) extends Tester(c)


.. chisel:attr:: class StackSpec extends ChiselPropSpec


Stop.scala
----------
.. chisel:attr:: class StopTester() extends BasicTester


.. chisel:attr:: class StopImmediatelyTester extends BasicTester


.. chisel:attr:: class StopSpec extends ChiselFlatSpec


EnableShiftRegister.scala
-------------------------
.. chisel:attr:: class EnableShiftRegister extends Module


.. chisel:attr:: class EnableShiftRegisterTester(c: EnableShiftRegister) extends Tester(c)


.. chisel:attr:: class EnableShiftRegisterSpec extends ChiselPropSpec


IOCompatibility.scala
---------------------
.. chisel:attr:: class IOCSimpleIO extends Bundle


.. chisel:attr:: class IOCPlusOne extends Module


.. chisel:attr:: class IOCModuleVec(val n: Int) extends Module


.. chisel:attr:: class IOCModuleWire extends Module


.. chisel:attr:: class IOCompatibilitySpec extends ChiselPropSpec with Matchers


TesterDriverSpec.scala
----------------------
.. chisel:attr:: class FinishTester extends BasicTester

	Extend BasicTester with a simple circuit and finish method.  TesterDriver will call the	finish method after the FinishTester's constructor has completed, which will alter the
	circuit after the constructor has finished.
  

	.. chisel:attr:: override def finish(): Unit =

	
		In finish we use last connect semantics to alter the test_wire in the circuit	with a new value
	    


.. chisel:attr:: class TesterDriverSpec extends ChiselFlatSpec


StrongEnum.scala
----------------
.. chisel:attr:: object EnumExample extends ChiselEnum


.. chisel:attr:: object OtherEnum extends ChiselEnum


.. chisel:attr:: object NonLiteralEnumType extends ChiselEnum


.. chisel:attr:: object NonIncreasingEnum extends ChiselEnum


.. chisel:attr:: class SimpleConnector(inType: Data, outType: Data) extends Module


.. chisel:attr:: class CastToUInt extends Module


.. chisel:attr:: class CastFromLit(in: UInt) extends Module


.. chisel:attr:: class CastFromNonLit extends Module


.. chisel:attr:: class CastFromNonLitWidth(w: Option[Int]


.. chisel:attr:: class EnumOps(val xType: ChiselEnum, val yType: ChiselEnum) extends Module


.. chisel:attr:: object StrongEnumFSM


.. chisel:attr:: class StrongEnumFSM extends Module


.. chisel:attr:: class CastToUIntTester extends BasicTester


.. chisel:attr:: class CastFromLitTester extends BasicTester


.. chisel:attr:: class CastFromNonLitTester extends BasicTester


.. chisel:attr:: class CastToInvalidEnumTester extends BasicTester


.. chisel:attr:: class EnumOpsTester extends BasicTester


.. chisel:attr:: class InvalidEnumOpsTester extends BasicTester


.. chisel:attr:: class IsLitTester extends BasicTester


.. chisel:attr:: class NextTester extends BasicTester


.. chisel:attr:: class WidthTester extends BasicTester


.. chisel:attr:: class StrongEnumFSMTester extends BasicTester


.. chisel:attr:: class StrongEnumSpec extends ChiselFlatSpec


.. chisel:attr:: class StrongEnumAnnotator extends Module


.. chisel:attr:: class StrongEnumAnnotatorWithChiselName extends Module


.. chisel:attr:: class StrongEnumAnnotationSpec extends FreeSpec with Matchers


LFSR16.scala
------------
.. chisel:attr:: class LFSRDistribution(gen: => UInt, cycles: Int = 10000) extends BasicTester

		This test creates two 4 sided dice.
	Each cycle it adds them together and adds a count to the bin corresponding to that value
	The asserts check that the bins show the correct distribution.
  

.. chisel:attr:: class LFSRDistribution(gen: => UInt, cycles: Int


.. chisel:attr:: class LFSRMaxPeriod(gen: => UInt) extends BasicTester


.. chisel:attr:: class MeetTheNewLFSR16SameAsTheOldLFSR16 extends BasicTester

	Check that the output of the new LFSR is the same as the old LFSR 


.. chisel:attr:: class LFSRSpec extends ChiselPropSpec


IntegerMathSpec.scala
---------------------
.. chisel:attr:: class IntegerMathTester extends BasicTester


.. chisel:attr:: class IntegerMathSpec extends ChiselPropSpec


ExtModule.scala
---------------
.. chisel:attr:: class ExtModuleTester extends BasicTester


.. chisel:attr:: class MultiExtModuleTester extends BasicTester

	Instantiate multiple BlackBoxes with similar interfaces but different	functionality. Used to detect failures in BlackBox naming and module
	deduplication.
  

.. chisel:attr:: class ExtModuleSpec extends ChiselFlatSpec


DontTouchSpec.scala
-------------------
.. chisel:attr:: class HasDeadCodeChild(withDontTouch: Boolean) extends Module


.. chisel:attr:: class HasDeadCode(withDontTouch: Boolean) extends Module


.. chisel:attr:: class DontTouchSpec extends ChiselFlatSpec


Printf.scala
------------
.. chisel:attr:: class SinglePrintfTester() extends BasicTester


.. chisel:attr:: class ASCIIPrintfTester() extends BasicTester


.. chisel:attr:: class MultiPrintfTester() extends BasicTester


.. chisel:attr:: class ASCIIPrintableTester extends BasicTester


.. chisel:attr:: class PrintfSpec extends ChiselFlatSpec


Mem.scala
---------
.. chisel:attr:: class MemVecTester extends BasicTester


.. chisel:attr:: class SyncReadMemTester extends BasicTester


.. chisel:attr:: class SyncReadMemWithZeroWidthTester extends BasicTester


.. chisel:attr:: class HugeSMemTester(size: BigInt) extends BasicTester


.. chisel:attr:: class HugeCMemTester(size: BigInt) extends BasicTester


.. chisel:attr:: class MemorySpec extends ChiselPropSpec


LoadMemoryFromFileSpec.scala
----------------------------
.. chisel:attr:: class UsesThreeMems(memoryDepth: Int, memoryType: Data) extends Module


.. chisel:attr:: class UsesMem(memoryDepth: Int, memoryType: Data) extends Module


.. chisel:attr:: class UsesMemLow(memoryDepth: Int, memoryType: Data) extends Module


.. chisel:attr:: class FileHasSuffix(memoryDepth: Int, memoryType: Data) extends Module


.. chisel:attr:: class MemoryShape extends Bundle


.. chisel:attr:: class HasComplexMemory(memoryDepth: Int) extends Module


.. chisel:attr:: class LoadMemoryFromFileSpec extends FreeSpec with Matchers

		The following tests are a bit incomplete and check that the output verilog is properly constructed
	For more complete working examples
	@see <a href="https://github.com/freechipsproject/chisel-testers">Chisel Testers</a> LoadMemoryFromFileSpec.scala
  

InstanceNameSpec.scala
----------------------
.. chisel:attr:: class InstanceNameModule extends Module


.. chisel:attr:: class InstanceNameSpec extends ChiselFlatSpec


RangeSpec.scala
---------------
.. chisel:attr:: class RangeSpec extends FreeSpec with Matchers


Math.scala
----------
.. chisel:attr:: class Math extends ChiselPropSpec


RebindingSpec.scala
-------------------
.. chisel:attr:: class RebindingSpec extends ChiselFlatSpec


SIntOps.scala
-------------
.. chisel:attr:: class SIntOps extends Module


.. chisel:attr:: class SIntOpsTester(c: SIntOps) extends Tester(c)


.. chisel:attr:: class SIntLitExtractTester extends BasicTester


.. chisel:attr:: class SIntOpsSpec extends ChiselPropSpec


OneHotMuxSpec.scala
-------------------
.. chisel:attr:: class OneHotMuxSpec extends FreeSpec with Matchers with ChiselRunners


.. chisel:attr:: class SimpleOneHotTester extends BasicTester


.. chisel:attr:: class SIntOneHotTester extends BasicTester


.. chisel:attr:: class FixedPointOneHotTester extends BasicTester


.. chisel:attr:: class AllSameFixedPointOneHotTester extends BasicTester


.. chisel:attr:: class ParameterizedOneHotTester[T <: Data](values: Seq[T], outGen: T, expected: T) extends BasicTester


.. chisel:attr:: class Agg1 extends Bundle


.. chisel:attr:: object Agg1 extends HasMakeLit[Agg1]


.. chisel:attr:: class Agg2 extends Bundle


.. chisel:attr:: object Agg2 extends HasMakeLit[Agg2]


.. chisel:attr:: class ParameterizedAggregateOneHotTester extends BasicTester


.. chisel:attr:: trait HasMakeLit[T]


.. chisel:attr:: class ParameterizedOneHot[T <: Data](values: Seq[T], outGen: T) extends Module


.. chisel:attr:: class ParameterizedAggregateOneHot[T <: Data](valGen: HasMakeLit[T], outGen: T) extends Module


.. chisel:attr:: class Bundle1 extends Bundle


.. chisel:attr:: class InferredWidthAggregateOneHotTester extends BasicTester


.. chisel:attr:: class Bundle2 extends Bundle


.. chisel:attr:: class Bundle3 extends Bundle


.. chisel:attr:: class DifferentBundleOneHotTester extends BasicTester


.. chisel:attr:: class UIntToOHTester extends BasicTester


Util.scala
----------
.. chisel:attr:: class PassthroughModuleIO extends Bundle


.. chisel:attr:: trait AbstractPassthroughModule extends RawModule


.. chisel:attr:: class PassthroughModule extends Module with AbstractPassthroughModule class PassthroughMultiIOModule extends MultiIOModule with AbstractPassthroughModule class PassthroughRawModule extends RawModule with AbstractPassthroughModule


.. chisel:attr:: class PassthroughMultiIOModule extends MultiIOModule with AbstractPassthroughModule class PassthroughRawModule extends RawModule with AbstractPassthroughModule    


.. chisel:attr:: class PassthroughRawModule extends RawModule with AbstractPassthroughModule    


Clock.scala
-----------
.. chisel:attr:: class ClockAsUIntTester extends BasicTester


.. chisel:attr:: class WithClockAndNoReset extends RawModule


.. chisel:attr:: class ClockSpec extends ChiselPropSpec


PopCount.scala
--------------
.. chisel:attr:: class PopCountTester(n: Int) extends BasicTester


.. chisel:attr:: class PopCountSpec extends ChiselPropSpec


OptionBundle.scala
------------------
.. chisel:attr:: class OptionBundle(val hasIn: Boolean) extends Bundle


.. chisel:attr:: class OptionBundleModule(val hasIn: Boolean) extends Module


.. chisel:attr:: class SomeOptionBundleTester(expected: Boolean) extends BasicTester


.. chisel:attr:: class NoneOptionBundleTester() extends BasicTester


.. chisel:attr:: class InvalidOptionBundleTester() extends BasicTester


.. chisel:attr:: class OptionBundleSpec extends ChiselFlatSpec


RawModuleSpec.scala
-------------------
.. chisel:attr:: class UnclockedPlusOne extends RawModule


.. chisel:attr:: class RawModuleTester extends BasicTester


.. chisel:attr:: class PlusOneModule extends Module


.. chisel:attr:: class RawModuleWithImplicitModule extends RawModule


.. chisel:attr:: class ImplicitModuleInRawModuleTester extends BasicTester


.. chisel:attr:: class RawModuleWithDirectImplicitModule extends RawModule


.. chisel:attr:: class ImplicitModuleDirectlyInRawModuleTester extends BasicTester


.. chisel:attr:: class RawModuleSpec extends ChiselFlatSpec


AnnotationNoDedup.scala
-----------------------
.. chisel:attr:: class MuchUsedModule extends Module


.. chisel:attr:: class UsesMuchUsedModule(addAnnos: Boolean) extends Module


.. chisel:attr:: class AnnotationNoDedup extends FreeSpec with Matchers


Harness.scala
-------------
.. chisel:attr:: class HarnessSpec extends ChiselPropSpec with BackendCompilationUtilities


	.. chisel:attr:: def simpleHarnessBackend(make: File => File): (File, String) =

	
		Compiles a C++ emulator from Verilog and returns the path to the	executable and the executable filename as a tuple.
	    


QueueSpec.scala
---------------
.. chisel:attr:: class ThingsPassThroughTester(elements: Seq[Int], queueDepth: Int, bitWidth: Int, tap: Int) extends BasicTester


.. chisel:attr:: class QueueReasonableReadyValid(elements: Seq[Int], queueDepth: Int, bitWidth: Int, tap: Int) extends BasicTester


.. chisel:attr:: class CountIsCorrectTester(elements: Seq[Int], queueDepth: Int, bitWidth: Int, tap: Int) extends BasicTester


.. chisel:attr:: class QueueSinglePipeTester(elements: Seq[Int], bitWidth: Int, tap: Int) extends BasicTester


.. chisel:attr:: class QueuePipeTester(elements: Seq[Int], queueDepth: Int, bitWidth: Int, tap: Int) extends BasicTester


.. chisel:attr:: class QueueFlowTester(elements: Seq[Int], queueDepth: Int, bitWidth: Int, tap: Int) extends BasicTester


.. chisel:attr:: class QueueSpec extends ChiselPropSpec


BlackBox.scala
--------------
.. chisel:attr:: class BlackBoxInverter extends BlackBox


.. chisel:attr:: class BlackBoxPassthrough extends BlackBox


.. chisel:attr:: class BlackBoxPassthrough2 extends BlackBox


.. chisel:attr:: class BlackBoxRegister extends BlackBox


.. chisel:attr:: class BlackBoxTester extends BasicTester


.. chisel:attr:: class BlackBoxFlipTester extends BasicTester


.. chisel:attr:: class MultiBlackBoxTester extends BasicTester

	Instantiate multiple BlackBoxes with similar interfaces but different	functionality. Used to detect failures in BlackBox naming and module
	deduplication.
  

.. chisel:attr:: class BlackBoxWithClockTester extends BasicTester


.. chisel:attr:: class BlackBoxConstant(value: Int) extends BlackBox(Map("VALUE" -> value, "WIDTH" -> log2Ceil(value + 1)))


.. chisel:attr:: class BlackBoxStringParam(str: String) extends BlackBox(Map("STRING" -> str))


.. chisel:attr:: class BlackBoxRealParam(dbl: Double) extends BlackBox(Map("REAL" -> dbl))


.. chisel:attr:: class BlackBoxTypeParam(w: Int, raw: String) extends BlackBox(Map("T" -> RawParam(raw)))


.. chisel:attr:: class BlackBoxWithParamsTester extends BasicTester


.. chisel:attr:: class BlackBoxSpec extends ChiselFlatSpec


EnumSpec.scala
--------------
.. chisel:attr:: class EnumSpec extends ChiselFlatSpec


AutoClonetypeSpec.scala
-----------------------
.. chisel:attr:: class BundleWithIntArg(val i: Int) extends Bundle


.. chisel:attr:: class BundleWithImplicit()(implicit val ii: Int) extends Bundle


.. chisel:attr:: class BundleWithArgAndImplicit(val i: Int)(implicit val ii: Int) extends Bundle


.. chisel:attr:: class BaseBundleVal(val i: Int) extends Bundle


.. chisel:attr:: class SubBundle(i: Int, val i2: Int) extends BaseBundleVal(i)


.. chisel:attr:: class SubBundleInvalid(i: Int, val i2: Int) extends BaseBundleVal(i + 1)


.. chisel:attr:: class BaseBundleNonVal(i: Int) extends Bundle


.. chisel:attr:: class SubBundleVal(val i: Int, val i2: Int) extends BaseBundleNonVal(i)


.. chisel:attr:: class ModuleWithInner extends Module


.. chisel:attr:: object CompanionObjectWithBundle


.. chisel:attr:: class NestedAnonymousBundle extends Bundle


.. chisel:attr:: class BundleWithArgumentField(val x: Data, val y: Data) extends Bundle  class AutoClonetypeSpec extends ChiselFlatSpec


.. chisel:attr:: class AutoClonetypeSpec extends ChiselFlatSpec


Tbl.scala
---------
.. chisel:attr:: class Tbl(w: Int, n: Int) extends Module


.. chisel:attr:: class TblTester(w: Int, n: Int, idxs: List[Int], values: List[Int]) extends BasicTester


.. chisel:attr:: class TblSpec extends ChiselPropSpec


NamingAnnotationTest.scala
--------------------------
.. chisel:attr:: trait NamedModuleTester extends MultiIOModule


	.. chisel:attr:: def expectName[T <: InstanceId](node: T, fullName: String): T =

	
		Expects some name for a node that is propagated to FIRRTL.	The node is returned allowing this to be called inline.
	    


	.. chisel:attr:: def expectModuleName[T <: Module](node: T, fullName: String): T =

	
		Expects some name for a module declaration that is propagated to FIRRTL.	The node is returned allowing this to be called inline.
	    


	.. chisel:attr:: def getNameFailures(): List[(InstanceId, String, String)] =

	
		After this module has been elaborated, returns a list of (node, expected name, actual name)	that did not match expectations.
		Returns an empty list if everything was fine.
	    


.. chisel:attr:: class NamedModule extends NamedModuleTester


.. chisel:attr:: class NameCollisionModule extends NamedModuleTester


.. chisel:attr:: class NonNamedModule extends NamedModuleTester

	Ensure no crash happens if a named function is enclosed in a non-named module  

.. chisel:attr:: object NonNamedHelper

	Ensure no crash happens if a named function is enclosed in a non-named function in a named	module.
  

.. chisel:attr:: class NonNamedFunction extends NamedModuleTester


.. chisel:attr:: class PartialNamedModule extends NamedModuleTester

	Ensure broken links in the chain are simply dropped  

.. chisel:attr:: class NamingAnnotationSpec extends ChiselPropSpec

	A simple test that checks the recursive function val naming annotation both compiles and	generates the expected names.
  

Counter.scala
-------------
.. chisel:attr:: class CountTester(max: Int) extends BasicTester


.. chisel:attr:: class EnableTester(seed: Int) extends BasicTester


.. chisel:attr:: class WrapTester(max: Int) extends BasicTester


.. chisel:attr:: class CounterSpec extends ChiselPropSpec


MemorySearch.scala
------------------
.. chisel:attr:: class MemorySearch extends Module


.. chisel:attr:: class MemorySearchTester(c: MemorySearch) extends Tester(c)


.. chisel:attr:: class MemorySearchSpec extends ChiselPropSpec


Module.scala
------------
.. chisel:attr:: class SimpleIO extends Bundle


.. chisel:attr:: class PlusOne extends Module


.. chisel:attr:: class ModuleVec(val n: Int) extends Module


.. chisel:attr:: class ModuleWire extends Module


.. chisel:attr:: class ModuleWhen extends Module


.. chisel:attr:: class ModuleForgetWrapper extends Module


.. chisel:attr:: class ModuleDoubleWrap extends Module


.. chisel:attr:: class ModuleRewrap extends Module


.. chisel:attr:: class ModuleWrapper(gen: => Module) extends Module


.. chisel:attr:: class NullModuleWrapper extends Module


.. chisel:attr:: class ModuleSpec extends ChiselPropSpec


AutoNestedCloneSpec.scala
-------------------------
.. chisel:attr:: class BundleWithAnonymousInner(val w: Int) extends Bundle


.. chisel:attr:: class AutoNestedCloneSpec extends ChiselFlatSpec with Matchers


MuxSpec.scala
-------------
.. chisel:attr:: class MuxTester extends BasicTester


.. chisel:attr:: class MuxSpec extends ChiselFlatSpec


GCD.scala
---------
.. chisel:attr:: class GCD extends Module


.. chisel:attr:: class GCDTester(a: Int, b: Int, z: Int) extends BasicTester


.. chisel:attr:: class GCDSpec extends ChiselPropSpec


Padding.scala
-------------
.. chisel:attr:: class Padder extends Module


.. chisel:attr:: class PadsTester(c: Pads) extends Tester(c)


.. chisel:attr:: class PadderSpec extends ChiselPropSpec


AnalogSpec.scala
----------------
.. chisel:attr:: class AnalogReaderIO extends Bundle


.. chisel:attr:: class AnalogWriterIO extends Bundle


.. chisel:attr:: trait AnalogReader


.. chisel:attr:: class AnalogReaderBlackBox extends BlackBox with AnalogReader


.. chisel:attr:: class AnalogReaderWrapper extends Module with AnalogReader


.. chisel:attr:: class AnalogWriterBlackBox extends BlackBox


.. chisel:attr:: class AnalogConnector extends Module


.. chisel:attr:: class VecAnalogReaderWrapper extends RawModule with AnalogReader


.. chisel:attr:: class VecBundleAnalogReaderWrapper extends RawModule with AnalogReader


.. chisel:attr:: abstract class AnalogTester extends BasicTester


.. chisel:attr:: class AnalogSpec extends ChiselFlatSpec



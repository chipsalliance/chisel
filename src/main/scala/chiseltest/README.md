# ChiselTest Compatibility Layer for Chisel 7

This directory contains a compatibility layer that enables existing code to use the **ChiselTest API** while running on **Chisel 7.5.0** with **ChiselSim**.

## Overview

The Chisel language underwent a major transition from Chisel 6 to Chisel 7:
- **Chisel 6** used the `ChiselTest` library for hardware simulation
- **Chisel 7** replaced it with `ChiselSim`, requiring all tests to be rewritten

This compatibility layer bridges that gap by providing the familiar ChiselTest API as a thin wrapper around ChiselSim, allowing existing code to run without modification.

## Key Features

✅ **Automatic Reset** - Just like ChiselTest in Chisel 6, modules are automatically reset before each test  
✅ **Drop-in Replacement** - Same imports, same API, works immediately  
✅ **Configurable** - Override `autoResetEnabled` or `resetCycles` to customize behavior

## Components

### 1. **package.scala**
Core compatibility layer providing implicit conversions and extension methods:

- **`testableData[T]`** - Adds poke/peek/expect methods to any Data type
- **`testableUInt`** - Specialized handling for UInt with `peekInt()` support
- **`testableBoolExt`** - Specialized handling for Bool with `peekBoolean()` support
- **`testableClock`** - Adds `step()` and `setTimeout()` methods to Clock
- **`testableReset`** - Adds poke support to Reset signals
- **`DecoupledIOOps[T]`** - Utilities for testing Decoupled interfaces (enqueueNow, expectDequeueNow, etc.)
- **Annotation stubs** - `WriteVcdAnnotation`, `VerilatorBackendAnnotation` for compatibility
- **`fork` function** - Stub implementation for concurrent test patterns

### 2. **ChiselScalatestTester.scala**
ScalaTest integration trait that provides the test runner:

- **`ChiselScalatestTester`** trait - Mix into your test class to enable `test()` method
- **Automatic reset** - Automatically applies reset before each test (like ChiselTest in Chisel 6)
- **`TestBuilder`** - Enables method chaining with `.withAnnotations()`
- **`TestRunner`** - Executes the actual test via `simulate()`
- **Configurable reset** - Override `autoResetEnabled` or `resetCycles` to customize behavior

### 3. **formal/package.scala**
Formal verification stubs (limited support):

- **`Formal`** trait - For formal verification test patterns
- **`BoundedCheck`** - Annotation for bounded model checking
- **`past()` function** - Temporal operator stub

## Usage

### Basic Test Pattern

```scala
import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class MyModuleTest extends AnyFlatSpec with ChiselScalatestTester {
  "MyModule" should "work" in {
    test(new MyModule) { dut =>
      // Module is automatically reset before this point (like ChiselTest)
      dut.io.input.poke(42.U)
      dut.clock.step()
      dut.io.output.expect(42.U)
    }
  }
}
```

### Disable Automatic Reset

If you need to control reset manually:

```scala
class MyModuleTest extends AnyFlatSpec with ChiselScalatestTester {
  // Disable automatic reset
  override def autoResetEnabled: Boolean = false
  
  "MyModule" should "work" in {
    test(new MyModule) { dut =>
      // Manually control reset
      dut.reset.poke(true.B)
      dut.clock.step()
      dut.reset.poke(false.B)
      
      dut.io.input.poke(42.U)
      dut.clock.step()
      dut.io.output.expect(42.U)
    }
  }
}
```

### Custom Reset Duration

```scala
class MyModuleTest extends AnyFlatSpec with ChiselScalatestTester {
  // Reset for 5 cycles instead of default 1
  override def resetCycles: Int = 5
  
  "MyModule" should "work" in {
    test(new MyModule) { dut =>
      // Module has been reset for 5 cycles
      dut.io.input.poke(42.U)
      dut.clock.step()
      dut.io.output.expect(42.U)
    }
  }
}
```

### With Annotations (Ignored but Supported)

```scala
test(new MyModule).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
  // Test code
}
```

### Decoupled Interface Testing

```scala
implicit val clk: Clock = dut.clock

dut.io.in.initSource()
dut.io.out.initSink()

dut.io.in.enqueueNow(10.U)
dut.io.out.expectDequeueNow(10.U)
```

## API Reference

### Poke, Peek, Expect

```scala
dut.io.signal.poke(42.U)        // Drive a signal
val value = dut.io.signal.peek()  // Read current value
dut.io.signal.expect(42.U)      // Assert expected value
```

### Type-Specific Methods

```scala
// UInt
dut.io.counter.peekInt()         // Peek as BigInt

// Bool
dut.io.flag.peekBoolean()        // Peek as Scala Boolean
```

### Clock Control

```scala
dut.clock.step()                // Step one cycle
dut.clock.step(10)              // Step 10 cycles
dut.clock.setTimeout(10000)     // Set timeout (ignored in ChiselSim)
```

### Decoupled Helpers

```scala
dut.io.in.enqueueNow(data)                    // Send one value
dut.io.in.enqueueSeq(Seq(v1, v2, v3))       // Send sequence
dut.io.out.expectDequeueNow(data)            // Expect one value
dut.io.out.expectDequeueSeq(Seq(v1, v2))    // Expect sequence
dut.io.in.initSource()                        // Initialize source side
dut.io.out.initSink()                         // Initialize sink side
```

## Testing Examples

See the test files in `src/test/scala/` for comprehensive examples. 

## Migration from Chisel 6

If you're migrating code from Chisel 6:

1. **Import statements** - Change `import chiseltest._` to use this compatibility layer
2. **Test structure** - Should be identical; no code changes needed
3. **Run tests** - Use `mill` or your build tool as before


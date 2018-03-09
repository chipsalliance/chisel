# Chisel Style Guide

The Chisel style guide reflects the [Google Java style
guide](http://https://google.github.io/styleguide/javaguide.html) and the [General Public Scala style
guide](http://docs.scala-lang.org/style/). Specific rules below are to clarify
the style used for the chisel3 repo and repos related to Chisel (Firrtl).

**Goal:** Readability and consistency are the main purposes of the style guide.
Writing your code so someone else (or yourself) can grok it later is important
to code health and quality.

## Filenames
The source file name consists of the case-sensitive name of the top-level class
it contains, plus ".scala".

## Packages

Package definitions must contain the full path to the package from scala. If
you create a subpackage, it should go in a subdirectory.

    package directory.name.to.get.you.to.your.source

As in Scala, packages follow the [Java package naming convention](https://google.github.io/styleguide/javaguide.html#s5.2.1-package-names).
Note that these guidelines call for all lowercase, no underscores.

```scala
// Do this
package hardware.chips.topsecret.masterplan

// Not this
package hardware.chips.veryObvious.bad_style
```

We also suggest you do not use chisel3 as a package, and especially do not use it
as the final (innermost) package.

```scala
// Don't do this
package hardware.chips.newchip.superfastcomponent.chisel3

// This will lead to instantiating package members like so:
val module = Module(new chisel3.FastModule)

// Which collides with the chisel namespace
import chisel3._
```

## Imports
Avoid wildcard ( ._ ) imports, with the exception of chisel3._
All other imports must call out used methods.
import chisel3._ must be first and separated from remaining imports with extra
blank line.

**Reason:** This makes it clear where methods are defined.

Remaining imports must be listed alphabetically.

```scala
import chisel3._

import the.other.thing.that.i.reference.inline
import the.other.things.that.i.reference.{ClassOne, ClassTwo}


val myInline = inline.MakeAnInline()
val myClassOne = new ClassOne
```

## Tests
Test classes are named starting with the name of the class they are testing, and
ending with "Test".
Test files must start with the name of the class you are testing and end with
"Test.scala".
Test files should reside in a subdirectory called "tests".
The tests package should be composed of the package class you are testing.

```scala
package class.under.test.class
package tests
```

## Comments
We use scaladoc to automatically generate documentation from the source code.

```scala
/** Multiple lines of ScalaDoc text are written here,
  * wrapped normally...
  */
public int method(String p1) { ... }
```

... or in this single-line example:

```scala
/** An especially short bit of Javadoc. */
```

Write documentation as if the person reading it knows more about Scala and
Chisel than you. If you find comments in the code consider breaking them up
into seperate methods.

## Module Classes and Instances

Modules can take different forms in Chisel. First, in the verilog sense where
you instance the module and then hook it up. In this case Module(new MyMod()) is
returning a reference to the module.

```scala
val myMod = Module(new MyMod())
myMod.io <> hookUp
```

Second, in a more programmatic inline style with factory methods. In this case
Queue is actually returning the part of the IO bundle representing the queue's
output. The factory method takes the input IO to the queue and an optional param
for depth.

```scala
val queueOut = Queue(queueIn, depth=10)
```

The latter can be used for composing multiple functions into a single line.

```scala
val queueOut = Queue(
  Arbitrate.byRoundRobin(
    Queue(a), // depth assumed to be 1
    Queue(b, depth=3),
    Queue(c, depth=4)
  ),
  depth=10
)
```

## Naming Conventions

Chisel follows the [Scala Naming Conventions](http://docs.scala-lang.org/style/naming-conventions.html).
In general, Chisel code should use CamelCase for naming (ie. the first letter
of each word is capitalized except sometimes the first word).

### Why CamelCase instead of Snake\_Case?

The compiler inserts underscores when splitting Chisel/FIRRTL aggregate types
into Verilog types. The compiler uses underscores to preserve the original
structure of the data in the resulting Verilog. Because of the special meaning
of underscores in Chisel-generated Verilog, their use in naming is **strongly**
discouraged.

Consider the following Chisel code:

```scala
val msg = Wire(new Bundle {
  val valid = Bool()
  val addr = UInt(32)
  val data = UInt(64)
})
val msg_rec = Wire(Bool())
```

Which compiles to the Verilog:

```verilog
wire  msg_valid;
wire [31:0] msg_addr;
wire [63:0] msg_data;
wire  msg_rec;
```

The Verilog maintains the structure of the original aggregate wire `msg`.
However, because we named another variable `msg_rec`, it appears in the Verilog
as if `msg` had 4 fields instead of its actual 3! If we instead follow the
lowerCamelCase for values naming convention, the resulting Verilog makes more
sense:

```scala
val msg = Wire(new Bundle {
  val valid = Bool()
  val addr = UInt(32)
  val data = UInt(64)
})
val msgRec = Wire(Bool())
```

And its resulting Verilog:

```verilog
wire  msg_valid;
wire [31:0] msg_addr;
wire [63:0] msg_data;
wire  msgRec;
```

Much better.

### Modules and Bundles (Classes, Traits, and Objects)

Modules are Scala classes and thus use UpperCamelCase.

```scala
class ModuleNamingExample extends Module {
  ...
}
```

Similarly, other classes (Chisel & Scala) should be UpperCamelCase as well.

```scala
trait UsefulScalaUtilities {
  def isEven(n: Int): Boolean = (n % 2) == 0
  def isOdd(n: Int): Boolean = !isEven(n)
}

class MyCustomBundle extends Bundle {
  ...
}
// Companion object to MyCustomBundle
object MyCustomBundle {
  ...
}

```

### Values and Methods

Values and methods should use lowerCamelCase. (Unless the value is a constant.)

```scala
val mySuperReg = Reg(init = 0.asUInt(32))
def myImportantMethod(a: UInt): Bool = a < 23.asUInt
```

### Constants

Unlike the Google Java style, constants use UpperCamelCase, which is in line
with the official [Scala Naming
Conventions](https://docs.scala-lang.org/style/naming-conventions.html).
Constants are final fields (val or object) whose contents are deeply immutable
and belong to a package object or an object. Examples:

```scala
// Constants
object Constants {
  val Number = 5
  val Names = "Ed" :: "Ann" :: Nil
  val Ages = Map("Ed" -> 35, "Ann" -> 32)
}

// Not constants
class NonConstantsInClass {
  val inClass: String = "in-class"
}

object nonConstantsInObject {
  var varString = "var-string"
  val mutableCollection: scala.collection.mutable.Set[String]
  val mutableElements = Set(mutable)
}
```

### UpperCamelCase vs. lowerCamelCase

There is more than one reasonable way to covert English prose into camel case.
We follow the convention defined in the [Google Java style
guide](https://google.github.io/styleguide/javaguide.html#s5.3-camel-case). The
potentially non-obvious rule being to treat acronymns as words for the purpose
of camel case.

Note that the casing of the original words is almost entirely disregarded.
Example:

Prose form     | UpperCamelCase | lowerCamelCase | Incorrect
:------------- | :------------- | :------------- | :------------
find GCD       | FindGcd        | findGcd        | ~~findGCD~~
state for FSM  | StateForFsm    | stateForFsm    | ~~stateForFSM~~
mock dut       | MockDut        | mockDut        | ~~MockDUT~~
FIFO Generator | FifoGenerator  | fifoGenerator  | ~~FIFOGenerator~~

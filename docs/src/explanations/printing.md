---
layout: docs
title:  "Printing"
section: "chisel3"
---

# Printing in Chisel

Chisel provides the `printf` function for debugging purposes. It comes in two flavors:

* [Scala-style](#scala-style)
* [C-style](#c-style)

### Scala-style

Chisel also supports printf in a style similar to [Scala's String Interpolation](http://docs.scala-lang.org/overviews/core/string-interpolation.html). Chisel provides a custom string interpolator `cf` which follows C-style format specifiers (see section [C-style](#c-style) below).

Note that the Scala s-interpolator is not supported in Chisel constructs and will throw an error:

```scala mdoc:invisible
import chisel3._
```

```scala mdoc:fail
class MyModule extends Module {
  val in = IO(Input(UInt(8.W)))
  printf(s"in = $in\n")
}
```

Instead, use Chisel's `cf` interpolator as in the following examples:

```scala mdoc:compile-only
val myUInt = 33.U
printf(cf"myUInt = $myUInt") // myUInt = 33
```

Note that when concatenating `cf"..."` strings, you need to start with a `cf"..."` string:

```scala mdoc:compile-only
// Does not interpolate the second string
val myUInt = 33.U
printf("my normal string" + cf"myUInt = $myUInt")
```

#### Simple formatting

Other formats are available as follows:

```scala mdoc:compile-only
val myUInt = 33.U
// Hexadecimal
printf(cf"myUInt = 0x$myUInt%x") // myUInt = 0x21
// Binary
printf(cf"myUInt = $myUInt%b") // myUInt = 100001
// Character
printf(cf"myUInt = $myUInt%c") // myUInt = !
```

#### Aggregate data-types

Chisel provides default custom "pretty-printing" for Vecs and Bundles. The default printing of a Vec is similar to printing a Seq or List in Scala while printing a Bundle is similar to printing a Scala Map.

```scala mdoc:compile-only
val myVec = VecInit(5.U, 10.U, 13.U)
printf(cf"myVec = $myVec") // myVec = Vec(5, 10, 13)

val myBundle = Wire(new Bundle {
  val foo = UInt()
  val bar = UInt()
})
myBundle.foo := 3.U
myBundle.bar := 11.U
printf(cf"myBundle = $myBundle") // myBundle = Bundle(a -> 3, b -> 11)
```

#### Custom Printing

Chisel also provides the ability to specify _custom_ printing for user-defined Bundles.

```scala mdoc:compile-only
class Message extends Bundle {
  val valid = Bool()
  val addr = UInt(32.W)
  val length = UInt(4.W)
  val data = UInt(64.W)
  override def toPrintable: Printable = {
    val char = Mux(valid, 'v'.U, '-'.U)
    cf"Message:\n" +
    cf"  valid  : $char%c\n" +
    cf"  addr   : $addr%x\n" +
    cf"  length : $length\n" +
    cf"  data   : $data%x\n"
  }
}

val myMessage = Wire(new Message)
myMessage.valid := true.B
myMessage.addr := "h1234".U
myMessage.length := 10.U
myMessage.data := "hdeadbeef".U

printf(cf"$myMessage")
```

Which prints the following:

```
Message:
  valid  : v
  addr   : 0x00001234
  length : 10
  data   : 0x00000000deadbeef
```

Notice the use of `+` between `cf` interpolated "strings". The results of `cf` interpolation can be concatenated by using the `+` operator.

### C-Style

Chisel provides `printf` in a similar style to its C namesake. It accepts a double-quoted format string and a variable number of arguments which will then be printed on rising clock edges. Chisel supports the following format specifiers:

| Format Specifier | Meaning |
| :-----: | :-----: |
| `%d` | decimal number |
| `%x` | hexadecimal number |
| `%b` | binary number |
| `%c` | 8-bit ASCII character |
| `%%` | literal percent |

It also supports a small set of escape characters:

| Escape Character | Meaning |
| :-----: | :-----: |
| `\n` | newline |
| `\t` | tab |
| `\"` | literal double quote |
| `\'` | literal single quote |
| `\\` | literal backslash |

Note that single quotes do not require escaping, but are legal to escape.

Thus printf can be used in a way very similar to how it is used in C:

```scala mdoc:compile-only
val myUInt = 32.U
printf("myUInt = %d", myUInt) // myUInt = 32
```

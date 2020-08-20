---
layout: docs
title:  "Printing"
section: "chisel3"
---
Chisel provides the `printf` function for debugging purposes. It comes in two flavors:

* [Scala-style](#scala-style)
* [C-style](#c-style)

### Scala-style

Chisel also supports printf in a style similar to [Scala's String Interpolation](http://docs.scala-lang.org/overviews/core/string-interpolation.html). Chisel provides a custom string interpolator `p` which can be used as follows:

```scala
val myUInt = 33.U
printf(p"myUInt = $myUInt") // myUInt = 33
```

Note that when concatenating `p"..."` strings, you need to start with a `p"..."` string:

```scala
// Does not interpolate the second string
val myUInt = 33.U
printf("my normal string" + p"myUInt = $myUInt")
```

#### Simple formatting

Other formats are available as follows:

```scala
// Hexadecimal
printf(p"myUInt = 0x${Hexadecimal(myUInt)}") // myUInt = 0x21
// Binary
printf(p"myUInt = ${Binary(myUInt)}") // myUInt = 100001
// Character
printf(p"myUInt = ${Character(myUInt)}") // myUInt = !
```

We recognize that the format specifiers are verbose, so we are working on a more concise syntax.

#### Aggregate data-types

Chisel provides default custom "pretty-printing" for Vecs and Bundles. The default printing of a Vec is similar to printing a Seq or List in Scala while printing a Bundle is similar to printing a Scala Map.

```scala
val myVec = Vec(5.U, 10.U, 13.U)
printf(p"myVec = $myVec") // myVec = Vec(5, 10, 13)

val myBundle = Wire(new Bundle {
  val foo = UInt()
  val bar = UInt()
})
myBundle.foo := 3.U
myBundle.bar := 11.U
printf(p"myBundle = $myBundle") // myBundle = Bundle(a -> 3, b -> 11)
```

#### Custom Printing

Chisel also provides the ability to specify _custom_ printing for user-defined Bundles.

```scala
class Message extends Bundle {
  val valid = Bool()
  val addr = UInt(32.W)
  val length = UInt(4.W)
  val data = UInt(64.W)
  override def toPrintable: Printable = {
    val char = Mux(valid, 'v'.U, '-'.U)
    p"Message:\n" +
    p"  valid  : ${Character(char)}\n" +
    p"  addr   : 0x${Hexadecimal(addr)}\n" +
    p"  length : $length\n" +
    p"  data   : 0x${Hexadecimal(data)}\n"
  }
}

val myMessage = Wire(new Message)
myMessage.valid := true.B
myMessage.addr := "h1234".U
myMessage.length := 10.U
myMessage.data := "hdeadbeef".U

printf(p"$myMessage")
```

Which prints the following:

```
Message:
  valid  : v
  addr   : 0x00001234
  length : 10
  data   : 0x00000000deadbeef
```

Notice the use of `+` between `p` interpolated "strings". The results of `p` interpolation can be concatenated by using the `+` operator. For more information, please see the documentation

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

```scala
val myUInt = 32.U
printf("myUInt = %d", myUInt) // myUInt = 32
```

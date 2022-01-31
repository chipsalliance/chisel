# Notes on the Compiler Plug-in

The Chisel plugin provides some operations that are too difficult, or not possbile,
to implement through regular Scala code. 

# This documentation is for developers working on chisel internals.

## Compiler plugin operations
These are the two things that the compile plugin does.

1. Automatically generates the `cloneType` methods of Bundle
2. Changes the underlying mechanics of the `Bundle`s `elements` method in a way 
that does not require the use of **reflection** 
3. Future work: Make having a Seq[Data] in a bundle be a compiler error. See "Detecting Bundles with Seq[Data]" below.

### 1. Generating `cloneType` method
As of  Mar 18, 2021, PR #1826, generating the `cloneType` method (1. above) is now the default behavior.
The cloneType method used to be a tricky thing to write for chisel developers.
For historical purposes, here is the flag was used to control that prior to full adoption.
```
-P:chiselplugin:useBundlePlugin
```

### 2. Changing `Bundle#elements` method

A `Bundle` has a default `elements` method that relies on **reflection**, which is slow and brittle, to access the list of
*fields* the bundle contains.
When enabled this second operation of the plugin examines
the `Bundle`s AST in order to determine the fields and then re-writes the underlying code of `elements`.
Technically, rewriting a lower level private method `_elementsImpl`.
It is expected that the using this feature will shortly become the default.

>The plugin should not be enabled for the `main` chisel3 project because of internal considerations.
> It is enabled for the `Test` section.

In the meantime, advanced users can try using the feature by adding the following flag to the scalac options in their
chisel projects.

```
-P:chiselplugin:buildElementAccessor
```

For example in an `build.sbt` file adding the line 
```
scalacOptions += "-P:chiselplugin:genBundleElements",
```
in the appropriate place.

## Future work
### Detecting Bundles with Seq[Data]
Trying to have a `val Seq[Data]` (as opposed to a `val Vec[Data]` in a `Bundle` is a run time error.
Here is a block of code that could be added to the plugin to detect this case at compile time (with some refinement in
the detection mechanism):
```scala
  if (member.isAccessor && typeIsSeqOfData(member.tpe) && !isIgnoreSeqInBundle(bundleSymbol)) {
    global.reporter.error(
      member.pos,
      s"Bundle.field ${bundleSymbol.name}.${member.name} cannot be a Seq[Data]. " +
        "Use Vec or MixedVec or mix in trait IgnoreSeqInBundle"
    )
  }
```
### Notes about working on the `_elementsImpl` generator for the plugin in `BundleComponent.scala`
In general the easiest way to develop and debug new code in the plugin is to use `println` statements.
Naively this can result in reams of text that can be very hard to look through.

What I found to be useful was creating some wrappers for `println` that only printed when the `Bundles` had a particular name pattern.
- Create a regular expression string in the `BundleComponent` class
- Add a printf wrapper name `show` that checks the `Bundle`'s name against the regex
- For recursive code in `getAllBundleFields` create a different wrapper `indentShow` that indents debug lines
- Sprinkle calls to these wrappers as needed for debugging

#### Bundle Regex
```scala
    val bundleNameDebugRegex = "MyBundle.*"
```
#### Add `show` wrapper
`show` should be inside `case bundle` block of the `transform` method in order to have access to the current `Bundle`

```scala
def show(string: => String): Unit = {
  if (bundle.symbol.name.toString.matches(bundleNameDebugRegex)) {
    println(string)
  }
}
```
#### Add `indentShow` wrapper
This method can be added into `BundleComponent.scala` in the `transform` method after `case Bundle`
Inside of `getAllBundleFields` I added the following code that indented for each recursion up the current
`Bundle`'s hierarchy.
```scala
def indentShow(s: => String): Unit = {
  val indentString = ("-" * depth) * 2 + ">  "
  s.split("\n").foreach { line =>
    show(indentString + line)
  }
}
```


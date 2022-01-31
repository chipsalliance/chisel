# Notes on the Compiler Plug-in

The Chisel plugin provides some operations that are too difficult, or not possbile,
to implement through regular Scala code. 
>This documentation is for developers working on chisel internals.

## Compiler plugin operations
These are the three things that the compile plugin does.

1. Automatically generates the `cloneType` methods of Bundles
2. The plugin adds a method `_usingPlugin: Boolean = true` to show step 1. was done.
3. Generates a function that returns list of the hardware fields in a bundle `_elementsImpl`
4. Future work: Make having a Seq[Data] in a bundle be a compiler error. See "Detecting Bundles with Seq[Data]" below.

### Generating `cloneType` method
As of  Mar 18, 2021, PR #1826, generating the `cloneType` method (1. above) is now the default behavior.
For historical purposes, here is the flag was used to control that prior to full adoption.
```
-P:chiselplugin:useBundlePlugin
```

### Generating the `_elementsImpl` method
The plugin does not by default generate the optimized (no-reflection based) `_elementsImpl` method.
Turn on this feature by adding the `scalac` flag
```
-P:chiselplugin:buildElementAccessor
```

#### Detecting Bundles with Seq[Data]
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


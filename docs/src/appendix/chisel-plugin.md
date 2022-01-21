# Notes on the Compiler Plug-in

The Chisel plugin provides some operations that are too difficult, or not possbile, 
to implement through regular Scala code.

## Compiler plugin operations
1. Automatically generates the `cloneType` methods of Bundles
2. Records that this plugin ran on a bundle by adding a method `_usingPlugin: Boolean = true`
3. Generates a function that returns list of the hardware fields in a bundle `_elementsImpl`

### Generating `cloneType` method
After a period of testing this is now on by default. This flag was used to control that prior to full adoption.
```
-P:chiselplugin:useBundlePlugin
```

### Generating the `_elementsImpl` method
For generating the elementsImpl method, the default is off.
Turn it on by using the scalac flag
```
-P:chiselplugin:buildElementAccessor
```

#### Detecting Bundles with Seq[Data]
Currently having this is a run time error
Here is a block of code that could be added (with some refinement in 
the detection mechanism)
```scala
  //TODO: The following code block creates a compiler error for bundle fields that are
  //      Seq[Data]. The test for this case is a bit too hacky so for now we will just
  //      elide these fields, and will be caught in Aggregate.scala at runtime
  if (member.isAccessor && typeIsSeqOfData(member.tpe) && !isIgnoreSeqInBundle(bundleSymbol)) {
    global.reporter.error(
      member.pos,
      s"Bundle.field ${bundleSymbol.name}.${member.name} cannot be a Seq[Data]. " +
        "Use Vec or MixedVec or mix in trait IgnoreSeqInBundle"
    )
  }
```
## Notes about working on the `_elementsImpl` generator for the plugin in `BundleComponent.scala`
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


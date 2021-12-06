# Chisel-plugin

See: [Jack's notes on plugins](https://users.scala-lang.org/t/overriding-class-method-in-compiler-plugin/7135) for
some good background on using compiler plugins

The plugin runs after the `typer` phase

## BundleComponent
We want to construct a new and better `elements` method for Bundles.
The constructed method will be a simple `LinkedHashMap` of `String` to `Data`
Prior to this `elements` used reflection to access the internal fields

### General approach
Run the `BundleComponent` with `BundlePhase` to translate fields in Bundles and Traits
into a fixed Map of field name to field `Data` type

### Problems
`Trait`'s fields do not follow the same pattern as `Bundles`s.


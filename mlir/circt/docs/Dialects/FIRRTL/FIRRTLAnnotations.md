# FIRRTL Annotations

The Scala FIRRTL Compiler (SFC) provides a mechanism to encode arbitrary
metadata and associate it with zero or more "things" in a FIRRTL circuit.  This
mechanism is an _Annotation_ and the association is described using one or more
_Targets_.  Annotations should be viewed as an extension to the FIRRTL IR
specification, and can greatly affect the meaning and interpretation of the IR.

Annotations are represented as a dictionary, with a "class" field which
describes which annotation it is, and a "target" field which represents the IR
object it is attached to. The annotation's class matches the name of a Java
class in the Scala Chisel/FIRRTL code base. Annotations may have arbitrary
additional fields attached. Some annotation classes extend other annotations,
which effectively means that the subclass annotation implies to effect of the
parent annotation.

Annotations are serializable to JSON and either live in a separate file (e.g.,
during the handoff between Chisel and the SFC) or are stored in-memory (e.g.,
during SFC-based compilation).  The SFC pass API requires that passes describe
which targets in the circuit they update.  SFC infrastructure then
automatically updates annotations so they are always synchronized with their
corresponding FIRRTL IR.

An example of an annotation is the `DontTouchAnnotation`, which can be used to
indicate to the compiler that a wire "foo" should not be optimized away.

```json
{
  "class":"firrtl.transforms.DontTouchAnnotation",
  "target""~MyCircuit|MyModule>foo"
}
```

Some annotations have more complex interactions with the IR. For example the
[BoringUtils](https://javadoc.io/doc/edu.berkeley.cs/chisel3_2.13/latest/chisel3/util/experimental/BoringUtils$.html)
provides FIRRTL with annotations which can be used to wire together any two
things across the module instance hierarchy.

## Motivation

Historically, annotations grew out of three choices in the design of FIRRTL IR:

1) FIRRTL IR is not extensible with user-defined IR nodes.
2) FIRRTL IR is not parameterized.
3) FIRRTL IR does not support in-IR attributes.

Annotations have then been used for all manner of extensions including:

1) Encoding SystemVerilog nodes into the IR using special printfs, an example of
   working around (1) above.
2) Setting the reset vector of different, identical CPU cores, an example of
   working around (2) above.
3) Encoding sources and sinks that should be wired together by an SFC pass, an
   example of (3) above.

## Targets

A circuit is described, stored, and optimized in a folded representation. For
example, there may be multiple instances of a module which will eventually
become multiple physical copies of that module on the die.

Targets are a mechanism to identify specific hardware in specific instances of
modules in a FIRRTL circuit.  A target consists of a circuit, a root module, an
optional instance hierarchy, and an optional reference. A target can only
identify hardware with a name, e.g., a circuit, module, instance, register,
wire, or node. References may further refer to specific fields or subindices in
aggregates. A target with no instance hierarchy is local. A target with an
instance hierarchy is non-local.

Targets use a shorthand syntax of the form:
```
target ::= “~” (circuit) (“|” (module) (“/” (instance) “:” (module) )* (“>” (ref) )?)?
```

A reference is a name inside a module and one or more qualifying tokens that
encode subfields (of a bundle) or subindices (of a vector):
```
reference ::= (name) ("[" (index) "]" | "." (field))*
```

Targets are specific enough to refer to any specific module in a folded,
unfolded, or partially folded representation.

To show some examples of what these look like, consider the following example
circuit. This consists of four instances of module `Baz`, two instances of
module `Bar`, and one instance of module `Foo`:

```firrtl
circuit Foo:
  module Foo:
    inst a of Bar
    inst b of Bar
  module Bar:
    inst c of Baz
    inst d of Baz
  module Baz:
    skip
```

| Folded Module   | Unfolded Modules  |
| --------------- | ----------------- |
| <img title="Folded Modules" src="includes/img/firrtl-folded-module.png"/> | <img title="Unfolded Modules" src="includes/img/firrtl-unfolded-module.png"/> |

Using targets (or multiple targets), any specific module, instance, or
combination of instances can be expressed. Some examples include:

| Target                                 | Description                                                |
| --------                               | -------------                                              |
| <code>~Foo</code>                      | refers to the whole circuit                                |
| <code>~Foo&#124;Foo</code>             | refers to the top module                                   |
| <code>~Foo&#124;Bar</code>             | refers to module `Bar` (or both instances of module `Bar`) |
| <code>~Foo&#124;Foo/a:Bar</code>       | refers just to one instance of module `Bar`                |
| <code>~Foo&#124;Foo/b:Bar/c:Baz</code> | refers to one instance of module `Baz`                     |
| <code>~Foo&#124;Bar/d:Baz</code>       | refers to two instances of module `Baz`                    |

If a target does not contain an instance path, it is a _local_ target.  A local
target points to all instances of a module.  If a target contains an instance
path, it is a _non-local_ target.  A non-local target _may_ not point to all
instances of a module.  Additionally, a non-local target may have an equivalent
local target representation.

## Inline Annotations

The MLIR FIRRTL compiler supports an inline format for annotations as an
extension to the FIRRTL syntax. These inline annotations are helpful for making
single-file annotated FIRRTL code. This is not supported by the Scala FIRRTL
compiler.

Inline annotations are attached to the `circuit`, and are JSON wrapped in `%[`
and `]`.

```firrtl
circuit Foo: %[[{"a":"a","target":"~Foo"}]]
  module Foo:
    skip
```

## Annotations in CIRCT

We plan to provide full support for annotations in CIRCT.  The FIRRTL dialect
current supports:

1) All non-local annotations can be parsed and applied to the correct circuit
   component.
2) Annotations, with and without references, are copied to the correct ground
   type in the `LowerTypes` pass.

Annotations can be parsed using the `--annotation-file` command line argument
to the `firtool` utility.  Alternatively, we provide a non-standard way of
encoding annotations in the FIRRTL IR textual representation.  We provide this
non-standard support primarily to make test writing easier.  As an example of
this, consider the following JSON annotation file:

```json
[
  {
    "target": "~Foo|Foo",
    "hello": "world"
  }
]
```

This can be equivalently, in CIRCT, expressed as:

```firrtl
circuit Foo: %[[{"target":"~Foo|Foo","hello":"world"}]]
  module Foo:
    skip
```

During parsing, annotations are "scattered" into the MLIR representation as
operation or port attributes.  As an example of this, the above parses into the
following MLIR representation:

```mlir
firrtl.circuit "Foo" {
  firrtl.module @Foo() attributes {annotations = [{hello = "world"}]} {
    firrtl.skip
  }
}
```

Targets without references have their targets stripped during scattering since
target information is redundant once annotation metadata is attached to the IR.
Targets with references have the reference portion of the target included in
the attribute.  The `LowerTypes` pass then uses this reference information to
attach annotation metadata to only the _lowered_ portion of a targeted circuit
component.

Annotations are expected to be fully removed via custom transforms, conversion
to other MLIR operations, or dropped. A warning will be emitted if there are
any unused annotations still in the circuit. For example, the `ModuleInliner`
pass removes `firrtl.passes.InlineAnnotation` by inlining annotated modules or
instances. JSON Annotations map to the builtin MLIR attributes. An annotation
is implemented using a DictionaryAttr, which holds the class, target, and any
annotation specific data.

## Annotations

Annotations here are written in their JSON format. A "reference target"
indicates that the annotation could target any object in the hierarchy,
although there may be further restrictions in the annotation.

### [AttributeAnnotation](https://javadoc.io/doc/edu.berkeley.cs/firrtl_2.13/latest/firrtl/AttributeAnnotation.html)

| Property    | Type   | Description                  |
| ----------  | ------ | ---------------------------- |
| class       | string | `firrtl.AttributeAnnotation` |
| target      | string | A reference target           |
| description | string | An attribute                 |

This annotation attaches SV attributes to a specified target. A reference
target must be a wire, node, reg, or module. This annotation doesn't prevent
optimizations so it's necessary to add dontTouch annotation if users want to
preseve the target.

Example:
```json
{
  "class": "firrtl.AttributeAnnotation",
  "target": "~Foo|Foo>r",
  "description": "debug = \"true\""
}
```

### BlackBox

| Property   | Type   | Description                  |
| ---------- | ------ | -------------                |
| class      | string | `firrtl.transforms.BlackBox` |
| target     | string | An ExtModule name target     |

This annotation is attached to any external module created from any of the
other blackbox annotations, such as `BlackBoxInlineAnno`. This is used when
generating metadata about external modules to distinguish generated modules.
This annotation is internal to the MLIR FIRRTL compiler.

Example:
```json
{
  "class": "firrtl.transforms.BlackBox",
  "target": "~Foo|Foo",
}
```

### [BlackBoxInlineAnno](https://javadoc.io/doc/edu.berkeley.cs/firrtl_2.13/latest/firrtl/transforms/BlackBoxInlineAnno.html)

| Property   | Type   | Description                            |
| ---------- | ------ | -------------                          |
| class      | string | `firrtl.transforms.BlackBoxInlineAnno` |
| target     | string | An ExtModule name target               |
| name       | string | A full path to a file                  |
| text       | string | Literal verilog code.                  |

Specifies the black box source code (`text`) inline. Generates a file with
the given `name` in the target directory.

Example:
```json
{
  "class": "firrtl.transforms.BlackBoxInlineAnno",
  "target": "~Foo|Foo",
  "name": "blackbox-inline.v",
  "text": "module ExtInline(); endmodule\n"
}
```

### [BlackBoxPathAnno](https://javadoc.io/doc/edu.berkeley.cs/firrtl_2.13/latest/firrtl/transforms/BlackBoxPathAnno.html)

| Property   | Type   | Description                          |
| ---------- | ------ | -------------                        |
| class      | string | `firrtl.transforms.BlackBoxPathAnno` |
| target     | string | An ExtModule name target             |
| path       | string | ModuleName target                    |

Specifies the file `path` as source code for the module. Copies the file
to the target directory.

Example:
```json
{
  "class": "firrtl.transforms.BlackBoxPathAnno",
  "target": "~Foo|Foo",
  "path": "myfile.v"
}
```

### [BlackBoxResourceFileNameAnno](https://javadoc.io/doc/edu.berkeley.cs/firrtl_2.13/latest/firrtl/transforms/BlackBoxResourceFileNameAnno.html)

| Property         | Type   | Description                              |
| ----------       | ------ | -------------                            |
| class            | string | `firrtl.transforms.BlackBoxFileNameAnno` |
| resourceFileName | string | Output filename                          |

Specifies the output file name for the list of black box source files that
is generated as a collateral of the pass.

Example:
```json
{
  "class": "firrtl.transforms.BlackBoxResourceFileNameAnno",
  "resourceFileName": "FileList.f"
}
```

### [BlackBoxTargetDirAnno](https://javadoc.io/doc/edu.berkeley.cs/firrtl_2.13/latest/firrtl/transforms/BlackBoxTargetDirAnno.html)

| Property   | Type   | Description                               |
| ---------- | ------ | -------------                             |
| class      | string | `firrtl.transforms.BlackBoxTargetDirAnno` |
| targetDir  | string | Output directory                          |

Overrides the target directory into which black box source files are
emitted.

Example:
```json
{
  "class": "firrtl.transforms.BlackBoxTargetDirAnno",
  "targetDir": "/tmp/circt/output"
}
```

### Convention

| Property   | Type   | Description                             |
| ---------- | ------ | --------------------------------------- |
| class      | string | `circt.ConventionAnnotation`            |
| convention | string | `scalarized`                            |
| target     | string | Reference target                        |

Specify the port convention for a module. The port convention controls how a
module's ports are transformed, and how that module can be instantiated, in the
output format.

The options are:
- `scalarized`: Convert aggregate ports (i.e. vector or bundles) into multiple
  ground-typed ports.

```json
{
  "class": "circt.ConventionAnnotation",
  "convention": "scalarized",
  "target": "~Foo|Bar/d:Baz"
}
```

### ElaborationArtefactsDirectory

| Property   | Type   | Description                                              |
| ---------- | ------ | -------------                                            |
| class      | string | `sifive.enterprise.firrtl.ElaborationArtefactsDirectory` |
| dirname    | string | The artifact output directory                            |

This annotation is used to indicate the output directory or artifacts generated
by the ElaborationArtefacts transform.

Example:
```json
{
  "class": "sifive.enterprise.firrtl.ElaborationArtefactsDirectory",
  "dirname": "output/artefacts"
}
```

### AddSeqMemPortAnnotation

| Property | Type    | Description                                                   |
| -------- | ------  | -------------                                                 |
| class    | string  | `sifive.enterprise.firrtl.AddSeqMemPortAnnotation`            |
| name     | string  | The name of the port to insert                                |
| input    | bool    | If true this is an input port, otherwise it is an output port |
| width    | integer | The width of the port                                         |

This annotation causes an extra port to be added to all SRAMs modules in the
DUT. The extra port is a regular module port of unsigned integer type with the
specified width. These extra ports are commonly used to implement SRAM features
not represented by the FIRRTL memory op, such as MBIST.  The added port will be
wired to the DUT, where it will be tied to 0.

Example:
```json
{
  "class":"sifive.enterprise.firrtl.AddSeqMemPortAnnotation",
  "name":"user_outputs",
  "input":false,
  "width":1
}
```

### AddSeqMemPortsFileAnnotation

| Property | Type   | Description                                             |
| -------- | ------ | -------------                                           |
| class    | string | `sifive.enterprise.firrtl.AddSeqMemPortsFileAnnotation` |
| filename | string | The filename to output to                               |

This annotation is used to emit metadata about the extra ports created by
`AddSeqMemPortAnnotation`.  This file is emitted relative to the
`MetadataDirAnnotation`. The file lists each SRAM and provides the mapping to
where it is in the hierarchy, and gives its IO prefix at the DUT top level.

```
0 -> Dut.submodule.sram0.sram0_ext
1 -> Dut.submodule.sram1.sram1_ext
```

Example:
```json
{
  "class":"sifive.enterprise.firrtl.AddSeqMemPortsFileAnnotation",
  "filename":"SRAMPorts.txt"
}
```

### [DocStringAnnotation](https://javadoc.io/doc/edu.berkeley.cs/firrtl_2.13/latest/firrtl/DocStringAnnotation.html)

| Property    | Type   | Description                  |
| ----------  | ------ | ---------------------------- |
| class       | string | `firrtl.DocStringAnnotation` |
| target      | string | A reference target           |
| description | string | An attribute                 |

This annotation attaches a comment to a specified target. A reference
target must be a wire, node, reg, or module. This annotation doesn't prevent
optimizations so it's necessary to add dontTouch annotation if users want to
preseve the target.

Example:
```json
{
  "class": "firrtl.DocStringAnnotation",
  "target": "~Foo|Foo>r",
  "description": "comment"
}
```

### [DontTouchAnnotation](https://javadoc.io/doc/edu.berkeley.cs/firrtl_2.13/latest/firrtl/transforms/DontTouchAnnotation.html)

| Property   | Type   | Description                             |
| ---------- | ------ | -------------                           |
| class      | string | `firrtl.transforms.DontTouchAnnotation` |
| target     | string | Reference target                        |

The `DontTouchAnnotation` prevents the removal of elements through
optimization. This annotation is an optimization barrier, for
example, it blocks constant propagation through it.
This annotation also ensures that the name of the object is
preserved, and not discarded or modified.

Example:
```json
{
  "class": "firrtl.transforms.DontTouchAnnotation",
  "target": "~Foo|Bar/d:Baz"
}
```

### [FlattenAnnotation](https://javadoc.io/doc/edu.berkeley.cs/firrtl_2.13/latest/firrtl/transforms/FlattenAnnotation.html)

| Property   | Type   | Description                           |
| ---------- | ------ | -------------                         |
| class      | string | `firrtl.transforms.FlattenAnnotation` |
| target     | string | Reference target                      |

Indicates that the target should be flattened, which means that child instances
will be recursively inlined.

Example:
```json
{
  "class": "firrtl.transforms.FlattenAnnotation",
  "target": "~Foo|Bar/d:Baz"
}
```

### FullAsyncResetAnnotation

| Property   | Type   | Description                                         |
| ---------- | ------ | -------------                                       |
| class      | string | `sifive.enterprise.firrtl.FullAsyncResetAnnotation` |
| target     | string | Reference target                                    |

Indicates that all reset-less registers which are children of the target will
have an asynchronous reset attached, with a reset value of 0.

A module targeted by this annotation is not allowed to reside in multiple
hierarchies.

Example:
```json
{
  "class": "sifive.enterprise.firrtl.FullAsyncResetAnnotation",
  "target": "~Foo|Bar/d:Baz"
}
```

### IgnoreFullAsyncResetAnnotation

| Property   | Type   | Description                                               |
| ---------- | ------ | -------------                                             |
| class      | string | `sifive.enterprise.firrtl.IgnoreFullAsyncResetAnnotation` |
| target     | string | Reference target                                          |

This annotation indicates that the target should be excluded from the
FullAsyncResetAnnotation of a parent module.

Example:
```json
{
  "class": "sifive.enterprise.firrtl.IgnoreFullAsyncResetAnnotation",
  "target": "~Foo|Bar/d:Baz"
}
```

### InjectDUTHierarchyAnnotation

| Property | Type   | Description                                             |
|----------|--------|---------------------------------------------------------|
| class    | string | `sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation` |
| name     | string | The name of the module containing original DUT logic    |

This annotation can be used to add an extra level of hierarchy in the design
under the DUT (indicated with a `MarkDUTAnnotation`).  All logic in the original
DUT will be moved into a module with the specified `name`.  This is typically
used in combination with `ExtractBlackBoxAnnotation` (or with passes that add
these annotations to extract components like clock gates or memories) to not
intermix the original DUT contents with extracted module instantiations.

This annotation should only appear zero or once.

Example:
``` json
{
  "class": "sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation",
  "name": "Logic"
}
```

### [InlineAnnotation](https://javadoc.io/doc/edu.berkeley.cs/firrtl_2.13/latest/firrtl/passes/InlineAnnotation.html)

| Property   | Type   | Description                      |
| ---------- | ------ | -------------                    |
| class      | string | `firrtl.passes.InlineAnnotation` |
| target     | string | Reference target                 |

Indicates that the target should be inlined.

Example:
```json
{
  "class": "firrtl.passes.InlineAnnotation",
  "target": "~Foo|Bar/d:Baz"
}
```
### MarkDUTAnnotation

| Property   | Type   | Description                                  |
| ---------- | ------ | -------------                                |
| class      | string | `sifive.enterprise.firrtl.MarkDUTAnnotation` |
| target     | string | Reference target                             |

This annotation is used to mark the top module of the device under test. This
can be used to distinguish modules in the test harness from modules in the DUT.

Example:
```json
{
  "class":"sifive.enterprise.firrtl.MarkDUTAnnotation",
  "target":"Core.Core"
}
```

### MetadataDirAnnotation

| Property   | Type   | Description                                      |
| ---------- | ------ | -------------                                    |
| class      | string | `sifive.enterprise.firrtl.MetadataDirAnnotation` |
| dirname    | string | The directory to place generated metadata in     |

This annotation is used to define the directory where metadata should be
emitted. When this annotation is not present, metadata will be emitted to the
"metadata" directory by default.

Example:
```json
{
  "class":"sifive.enterprise.firrtl.MetadataDirAnnotation",
  "dirname":"build/metadata"
}
```

### ModuleHierarchyAnnotation

| Property   | Type   | Description                                          |
| ---------- | ------ | -------------                                        |
| class      | string | `sifive.enterprise.firrtl.ModuleHierarchyAnnotation` |
| filename   | string | The full output file path.                           |

This annotation indicates that a module hierarchy JSON file should be emitted
for the module hierarchy rooted at the design under test (DUT), as indicated by
the `MarkDUTAnnotation`. See the SV attribute, `firrtl.moduleHierarchyFile`, for
information about the JSON file format.

Example:
```json
{
  "class": "sifive.enterprise.firrtl.ModuleHierarchyAnnotation",
  "filename": "./dir/hier.json"
}
```

### MustDeduplicateAnnotation

| Property   | Type   | Description                                      |
| ---------- | ------ | -------------                                    |
| class      | string | `firrtl.transforms.MustDeduplicateAnnotation`    |
| modules    | array  | A list of module targets which must deduplicate. |

This annotation causes the deduplication pass to check that the listed modules
are deduplicated with each other.

Example:
```json
{
  "class":"firrtl.transforms.MustDeduplicateAnnotation",
  "modules":[
    "~Top|A"
    "~Top|B"
  ]
}
```

### NestedPrefixModulesAnnotation

| Property   | Type   | Description                                              |
| ---------- | ------ | -------------                                            |
| class      | string | `sifive.enterprise.firrtl.NestedPrefixModulesAnnotation` |
| prefix     | string | Prefix to use                                            |
| inclusive  | bool   | Whether this prefix is inclusive of the target           |

This annotations prefixes all module names under the target with the required
prefix.  If `inclusive` is true, it includes the target module in the renaming.
If `inclusive` is false, it will only rename modules instantiated underneath
the target module.  If a module is required to have two different prefixes, it
will be cloned.

This annotation is also applied to any interfaces or modules generated by the
Grand Central Views/Interfaces pass.  This annotation is applied _before_
`PrefixInterfacesAnnotation`.

Example:
```json
{
  "class": "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
  "prefix": "MyPrefix_",
  "inclusive": true
}
```

### OMIRFileAnnotation

| Property   | Type   | Description                                           |
| ---------- | ------ | -------------                                         |
| class      | string | `freechips.rocketchip.objectmodel.OMIRFileAnnotation` |
| filename   | string | Output file to emit OMIR to                           |

This annotation defines the output file to write the JSON-serialized OMIR to after compilation.

Example:
```json
{
  "class": "freechips.rocketchip.objectmodel.OMIRFileAnnotation",
  "filename": "path/to/omir.json"
}
```

### OMIRAnnotation

| Property   | Type   | Description                                       |
| ---------- | ------ | -------------                                     |
| class      | string | `freechips.rocketchip.objectmodel.OMIRAnnotation` |
| nodes      | array  | A list of OMIR nodes                              |

This annotation specifies a piece of Object Model 2.0 IR. The `nodes` field
is an array of individual OMIR nodes (Scala class `OMNode`), which have the
following form:
```json
{
  "info": "@[FileA line:col FileB line:col ...]",
  "id": "OMID:42",
  "fields": [/*...*/]
}
```
The `fields` entry is an array of individual OMIR fields (Scala class `OMField`), which have the following form:
```json
{
  "info": "@[FileA line:col FileB line:col ...]",
  "name": "foo",
  "value": /*...*/
}
```
The `value` field can be a JSON array or dictionary (corresponding to the `OMArray` and `OMMap` Scala classes, respectively), or any of the string-encoded OMIR classes:

- `OMMap:<fields>`
- `OMArray:<elements>`
- `OMReference:<id>`
- `OMBigInt:<value>`
- `OMInt:<value>`
- `OMLong:<value>`
- `OMString:<value>`
- `OMBoolean:<value>`
- `OMDouble:<value>`
- `OMBigDecimal:<value>`
- `OMFrozenTarget:<omir>`
- `OMDeleted`
- `OMConstant:<literal>`
- `OMReferenceTarget:<target>`
- `OMMemberReferenceTarget:<target>`
- `OMMemberInstanceTarget:<target>`
- `OMInstanceTarget:<target>`
- `OMDontTouchedReferenceTarget:<target>`

Example:
```json
{
  "class": "freechips.rocketchip.objectmodel.OMIRAnnotation",
  "nodes": [
    {
      "info": "",
      "id": "OMID:0",
      "fields": [
        {"info": "", "name": "a", "value": "OMReference:0"},
        {"info": "", "name": "b", "value": "OMBigInt:42"},
        {"info": "", "name": "c", "value": "OMLong:ff"},
        {"info": "", "name": "d", "value": "OMString:hello"},
        {"info": "", "name": "f", "value": "OMBigDecimal:10.5"},
        {"info": "", "name": "g", "value": "OMDeleted:"},
        {"info": "", "name": "h", "value": "OMConstant:UInt<2>(\"h1\")"},
        {"info": "", "name": "i", "value": 42},
        {"info": "", "name": "j", "value": true},
        {"info": "", "name": "k", "value": 3.14}
      ]
    },
    {
      "info": "",
      "id": "OMID:1",
      "fields": [
        {"info": "", "name": "a", "value": "OMReferenceTarget:~Foo|Foo"},
        {"info": "", "name": "b", "value": "OMInstanceTarget:~Foo|Foo"},
        {"info": "", "name": "c", "value": "OMMemberReferenceTarget:~Foo|Foo"},
        {"info": "", "name": "d", "value": "OMMemberInstanceTarget:~Foo|Foo"},
        {"info": "", "name": "e", "value": "OMDontTouchedReferenceTarget:~Foo|Foo"},
        {"info": "", "name": "f", "value": "OMReferenceTarget:~Foo|Bar"}
      ]
    }
  ]
}
```

### RetimeModuleAnnotation

| Property   | Type   | Description                                              |
| ---------- | ------ | -------------                                            |
| class      | string | `freechips.rocketchip.util.RetimeModuleAnnotation` |

This annotation is used to mark modules which should be retimed, and is
generally just passed through to other tools.

Example:
```json
{
    "class": "freechips.rocketchip.util.RetimeModuleAnnotation"
}
```

### RetimeModulesAnnotation

| Property   | Type   | Description                                              |
| ---------- | ------ | -------------                                            |
| class      | string | `sifive.enterprise.firrtl.RetimeModulesAnnotation` |
| filename   | string | The filename with full path where it will be written     |

This annotation triggers the creation of a file containing a JSON array
containing the names of all modules annotated with the
`RetimeModuleAnnotation`.

Example:
```json
{
  "class": "sifive.enterprise.firrtl.RetimeModuleAnnotation",
  "filename": "retime_modules.json"
}
```

### SeqMemInstanceMetadataAnnotation

| Property   | Type   | Description                                  								|
| ---------- | ------ | -------------                                								|
| class      | string | `sifive.enterprise.firrtl.SeqMemInstanceMetadataAnnotation` |
| target     | string | Reference target                             								|

This annotation attaches metadata to the firrtl.mem operation. The `data` is
emitted onto the `seq_mems.json` file. It is required for verification only and
used by memory generator tools for simulation.

Example:
```json
{
    "class":"sifive.enterprise.firrtl.SeqMemInstanceMetadataAnnotation",
    "data":{
      "baseAddress":2147483648,
      "eccScheme":"none",
      "eccBits":0,
      "dataBits":8,
      "eccIndices":[ ]
    },
    "target":"~CoreIPSubsystemVerifTestHarness|TLRAM>mem"
}
```

### ScalaClassAnnotation

| Property   | Type   | Description                                     |
| ---------- | ------ | -------------                                   |
| class      | string | `sifive.enterprise.firrtl.ScalaClassAnnotation` |
| target     | string | Reference target                                |
| className  | string | The corresponding class name                    |

This annotation records the name of the Java or Scala class which corresponds
to the module.

Example:
```json
{
  "class":"sifive.enterprise.firrtl.ScalaClassAnnotation",
  "target":"Top.ClockGroupAggregator",
  "className":"freechips.rocketchip.prci.ClockGroupAggregator"
}
```

### circt.Intrinsic

| Property   | Type   | Description       |
| ---------- | ------ | -------------     |
| class      | string | `circt.Intrinsic` |
| target     | string | Reference target  |
| intrinsic  | string | Name of Intrinsic |

Used to indicate an external module is really an intrinsic module.  This exists
to allow a frontend to generate intrinsics without FIRRTL language support for
intrinsics.  It is expected this will be deprecated as soon as the FIRRTL language
supports intrinsics.  This annotation can only be local and applied to a module.

### SitestBlackBoxAnnotation

| Property   | Type   | Description                                         |
| ---------- | ------ | -------------                                       |
| class      | string | `sifive.enterprise.firrtl.SitestBlackBoxAnnotation` |
| filename   | string | The file to write to                                |

This annotation triggers the creation of a file containing a JSON array of the
names of all external modules in the device under test which are not imported
or inlined blackbox modules. This will only collect modules which are
instantiated under a module annotated with `MarkDUTAnnotation`.

Example:
```json
{
  "class":"sifive.enterprise.firrtl.SitestBlackBoxAnnotation",
  "filename":"./blackboxes.json"
}
```

### SitestTestHarnessBlackBoxAnnotation

| Property   | Type   | Description                                                     |
| ---------- | ------ | -------------                                                   |
| class      | string | `sifive.enterprise.firrtl.SittestTestHarnessBlackBoxAnnotation` |
| filename   | string | The file to write to                                            |

This annotation triggers the creation of a file containing a JSON array of the
names of all external modules in the test harness which are not imported or
inlined blackbox modules. This will only collect modules which are not
instantiated under a module annotated with `MarkDUTAnnotation`.

Example:
```json
{
  "class":"sifive.enterprise.firrtl.SitestTestHarnessBlackBoxAnnotation",
  "filename":"./testharness_blackboxes.json"
}
```

### SubCircuitsTargetDirectory

| Property   | Type   | Description                                                        |
| ---------- | ------ | -------------                                                      |
| class      | string | `sifive.enterprise.grandcentral.phases.SubCircuitsTargetDirectory` |
| dir        | string | The sub-circuit output directory                                   |

This annotation is used to indicate the directory to serialize sub-circuits to
by GrandCentral. Sub-circuits will be put in subdirectories of `dir`, named by
their `circuitPackage` field.

In the Scala FIRRTL compiler this is attached to the circuit with the
commandline option `sub-circuits-target-dir`.
```
-sub-circuit-targets-dir <dir>
-sctd <dir>
```

Example:
```json
{
  "class":"sifive.enterprise.grandcentral.phases.SubCircuitsTargetDirectory",
  "dir":"verilog/verif.subcircuits"
}
```

### TestBenchDirAnnotation

| Property   | Type   | Description                                       |
| ---------- | ------ | -------------                                     |
| class      | string | `sifive.enterprise.firrtl.TestBenchDirAnnotation` |
| dirname    | string | The output directory                              |

This annotation is used to indicate where to emit the test bench modules
generated by GrandCentral.

Example:
```json
{
  "class": "sifive.enterprise.firrtl.TestBenchDirAnnotation",
  "dirname": "output/testbench"
}
```

### TestHarnessHierarchyAnnotation

| Property   | Type   | Description                                               |
| ---------- | ------ | -------------                                             |
| class      | string | `sifive.enterprise.firrtl.TestHarnessHierarchyAnnotation` |
| filename   | string | The full output file path.                                |

This annotation indicates that a module hierarchy JSON file should be emitted
for the module hierarchy rooted at the circuit root module, which is assumed to
be the test harness. See the SV attribute, `firrtl.moduleHierarchyFile`, for
information about the JSON file format.

Example:
```json
{
  "class": "sifive.enterprise.firrtl.TestHarnessHierarchyAnnotation",
  "filename": "./dir/hier.json"
}
```

### Instance Extraction

#### ExtractBlackBoxAnnotation

| Property    | Type   | Description                                                                 |
| --------    | ----   | -----------                                                                 |
| class       | string | `sifive.enterprise.firrtl.ExtractBlackBoxAnnotation`                        |
| target      | string | Reference target to the instance to be extracted                            |
| filename    | string | Output file to be filled with the applied hierarchy changes                 |
| prefix      | string | Prefix for the extracted instance                                           |
| dest        | string | Name of an optional wrapper module under which to group extracted instances |

This annotation causes the `ExtractInstances` pass to move the annotated
instance, or all instances if the annotation is on a module, upwards in the
hierarchy. If the `dest` field is present and non-empty, the instances are
placed in a module underneath the DUT (marked by `MarkDUTAnnotation`) with the
name provided in that field. If the `dest` field is empty, the instances are
extracted out of the DUT, such that the DUT gains additional ports that
correspond to the extracted instance ports. This allows the DUT to be
instantiated and custom implementations for the extracted instances to be
provided at the instantiation site. Instances are never extracted out of the
root module of the design.

Applies to modules and instances.

Example:
```json
{
  "class": "sifive.enterprise.firrtl.ExtractBlackBoxAnnotation",
  "target": "~TestHarness|MyBlackBox",
  "filename": "BlackBoxes.txt",
  "prefix": "bb",
  "dest": "BlackBoxes" // optional
}
```

#### ExtractClockGatesFileAnnotation

| Property    | Type   | Description                                                                 |
| --------    | ----   | -----------                                                                 |
| class       | string | `sifive.enterprise.firrtl.ExtractClockGatesFileAnnotation`                  |
| filename    | string | Output file to be filled with the applied hierarchy changes                 |
| group       | string | Name of an optional wrapper module under which to group extracted instances |

This annotation causes the `ExtractInstances` pass to move instances of
extmodules with defname `EICG_wrapper` upwards in the hierarchy, either out of
the DUT if `group` is omitted or empty, or into a submodule of the DUT with the
name given in `group`. The wiring prefix is hard-coded to `clock_gate`.

Applies to the circuit.

Example:
```json
{
  "class": "sifive.enterprise.firrtl.ExtractClockGatesFileAnnotation",
  "filename": "ClockGates.txt",
  "group": "ClockGates" // optional
}
```

#### ExtractSeqMemsFileAnnotation

| Property    | Type   | Description                                                                 |
| --------    | ----   | -----------                                                                 |
| class       | string | `sifive.enterprise.firrtl.ExtractSeqMemsFileAnnotation`                  |
| filename    | string | Output file to be filled with the applied hierarchy changes                 |
| group       | string | Name of an optional wrapper module under which to group extracted instances |

This annotation causes the `ExtractInstances` pass to move memory instances
upwards in the hierarchy, either out of the DUT if `group` is omitted or empty,
or into a submodule of the DUT with the name given in `group`. The wiring
prefix is hard-coded to `mem_wiring`.

Applies to the circuit.

Example:
```json
{
  "class": "sifive.enterprise.firrtl.ExtractSeqMemsFileAnnotation",
  "filename": "SeqMems.txt",
  "group": "SeqMems" // optional
}
```

## FIRRTL specific attributes applied to HW Modules

### Design Under Test

| Property   | Type   | Description                                   |
| ---------- | ------ | -------------                                 |
| class      | string | `sifive.enterprise.firrtl.MarkDUTAnnotation`  |
| target     | string | Reference target                              |

Marks what is the DUT (and not the testbench). This annotation is lowered to the
attribute `firrtl.DesignUnderTest` to indicate the module which is the DUT.

##  Grand Central

Grand Central provides annotations for creating cross module references and
SystemVerilog interfaces.

### Views

Grand Central views are used from Chisel to allow users to encapsulate monitor
logic that gets emitted separately from the DUT. The generated interfaces
provide a stable view of modules which are connected to the target module
through SystemVerilog bind statements.

#### TargetToken$Field

| Property   | Type              | Description                            |
| ---------- | ------            | -------------                          |
| class      | string            | `firrtl.annotations.TargetToken$Field` |
| value      | string or integer | Index or element name                  |

This is used to represent an index in to an aggregate type, such as an index or
array.

#### ReferenceTarget

| Property   | Type   | Description                                              |
| ---------- | ------ | -------------                                            |
| circuit    | string | Name of the encapsulating circuit                        |
| module     | string | Name of the root module of this reference                |
| path       | array  | Path through instance and Modules                        |
| ref        | string | Name of the component                                    |
| component  | array  | List of TargetToken$Field subcomponent of this reference |

A reference target is a JSON serialization of a regular reference target
string.

#### UnknownGroundType

| Property   | Type   | Description                                                          |
| ---------- | ------ | -------------                                                        |
| class      | string | `sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$` |

This represents an unknown FIRRTL ground type.

#### AugmentedGroundType

| Property   | Type   | Description                                          |
| ---------- | ------ | -------------                                        |
| class      | string | `sifive.enterprise.grandcentral.AugmentedGroundType` |
| ref        | object | ReferenceTarget of the target component              |
| tpe        | object | UnknownGroundType                                    |

Creates a SystemVerilog logic type.

Example:
```json
{
  "class": "sifive.enterprise.grandcentral.AugmentedGroundType",
  "ref": {
    "circuit": "GCTInterface",
    "module": "GCTInterface",
    "path": [],
    "ref": "a",
    "component": [
      {
        "class": "firrtl.annotations.TargetToken$Field",
        "value": "_2"
      },
      {
        "class": "firrtl.annotations.TargetToken$Index",
        "value": 0
      }
    ]
  },
  "tpe": {
    "class": "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"
  }
}
```

#### AugmentedVectorType

| Property   | Type   | Description                                          |
| ---------- | ------ | -------------                                        |
| class      | string | `sifive.enterprise.grandcentral.AugmentedVectorType` |
| elements   | array  | List of augmented types.

Creates a SystemVerilog unpacked array.

Example:
```json
{
  "class": "sifive.enterprise.grandcentral.AugmentedVectorType",
  "elements": [
    {
      "class": "sifive.enterprise.grandcentral.AugmentedGroundType",
      "ref": {
        "circuit": "GCTInterface",
        "module": "GCTInterface",
        "path": [],
        "ref": "a",
        "component": []
      },
      "tpe": {
        "class": "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"
      }
    },
    {
      "class": "sifive.enterprise.grandcentral.AugmentedGroundType",
      "ref": {
        "circuit": "GCTInterface",
        "module": "GCTInterface",
        "path": [],
        "ref": "b",
        "component": []
      },
      "tpe": {
        "class": "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"
      }
    }
  ]
}
```

#### AugmentedField

| Property    | Type   | Description                        |
| ----------  | ------ | -------------                      |
| name        | string | Name of the field                  |
| description | string | A textual description of this type |
| tpe         | string | A nested augmented type            |

A field in an augmented bundle type.  This can provide a small description of
what the field in the bundle is.

#### AugmentedBundleType

| Property   | Type   | Description                                        |
| ---------- | ------ | -------------                                      |
| class      | string | sifive.enterprise.grandcentral.AugmentedBundleType |
| defName    | string | The name of the SystemVerilog interface            |
| elements   | array  | List of AugmentedFields                            |

Creates a SystemVerilog interface for each bundle type.

#### ViewAnnotation, GrandCentralView$SerializedViewAnnotation

| Property    | Type     | Description                                                              |
| ----------- | -------- | ------------------                                                       |
| class       | string   | sifive.enterprise.grandcentral.GrandCentralView$SerializedViewAnnotation |
| name        | string   | Name of the view, no affect on output                                    |
| companion   | string   | Module target of an empty module to insert cross module references in to |
| parent      | string   | Module target of the module the interface will be referencing            |
| view        | object   | AugmentedBundleType representing the interface                           |

These annotations (which are equivalent) are used to represent a SystemVerilog
interface, a location in which it should be instantiated, and XMRs to drive the
interface.  Any XMR sources receive `DontTouchAnnotation` to prevent these from
being inadvertently deleted.  Note: this currently differs from the SFC
implementation where constant propagation is not supposed to be blocked by an
XMR.  Instead the source should be promoted to a literal value and driven on the
interface.

Either `ViewAnnotation` or `GrandCentralView$SerializedViewAnnotation` are the
same in CIRCT.  The latter, has its "view" value serialized (again) to JSON and
string-escaped.  When CIRCT sees any JSON string it tries to recursively
deserialize it.  If this fails, this is deemed to be a string.  If this
succeeds, then the JSON is unpacked.

The reason for this double serialization is due to a quirk of the JSON library
that the SFC uses.  This JSON library uses a type class pattern for users to
tell it how to deserialize custom types.  Because the `ViewAnnotation`lives in a
SiFive library, there is no mechanism to provide a type class implementation to
the function that does annotation deserialization inside the SFC.  Doubly
serializing enables the deserialization to be delayed until SFC Grand Central
passes run and a type class implementation is available.

Example:
```json
{
  "class": "sifive.enterprise.grandcentral.GrandCentralView$SerializedViewAnnotation",
  "name": "view",
  "companion": "~GCTInterface|view_companion",
  "parent": "~GCTInterface|GCTInterface",
  "view": {
    "class": "sifive.enterprise.grandcentral.AugmentedBundleType",
    "defName": "ViewName",
    "elements": [
      {
        "name": "port",
        "description": "the port 'a' in GCTInterface",
        "tpe": {
          "class": "sifive.enterprise.grandcentral.AugmentedGroundType",
          "ref": {
            "circuit": "GCTInterface",
            "module": "GCTInterface",
            "path": [],
            "ref": "a",
            "component": []
          },
          "tpe": {
            "class": "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"
          }
        }
      }
    ]
  }
}
```

#### ExtractGrandCentralAnnotation

| Property  | Type   | Description                                                  |
| ---       | ---    | ---                                                          |
| class     | string | sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation |
| directory | string | Directory where Grand Central outputs go, except a bindfile  |
| filename  | string | Filename with full path where the bindfile will be written   |

This annotation controls where to "extract" Grand Central collateral from the
circuit.  This annotation is mandatory and can only appear once if the full
Grand Central transform pipeline is run in the SFC.  (An error is generated by
the `ExtractGrandCentralCode` transform.)

The directory member has no effect on the filename member, i.e., the directory
will not be prepended to the filename.

Example:
```json
{
  "class": "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
  "directory": "gct-dir",
  "filename": "gct-dir/bindings.sv"
}
```

#### PrefixInterfacesAnnotation

| Property | Type   | Description                                               |
|----------|--------|-----------------------------------------------------------|
| class    | string | sifive.enterprise.grandcentral.PrefixInterfacesAnnotation |
| prefix   | string | A prefix to apply to all interface names                  |

This annotation can be used to set a global prefix for all interfaces generated
by Grand Central, including nested interfaces.  The prefix will be applied
_after_ any prefixes set by `NestedPrefixModulesAnnotation`.

This annotation may only exist zero or one times.  This differs from the SFC
implementation which will choose the first instance of this annotation.

Example:

``` json
{
  "class": "sifive.enterprise.grandcentral.PrefixInterfacesAnnotation",
  "prefix": "PREFIX_"
}
```

#### GrandCentralHierarchyFileAnnotation

| Property | Type   | Description                                                              |
|----------|--------|--------------------------------------------------------------------------|
| class    | string | sifive.enterprise.grandcentral.GrandCentralHierarchyFileAnnotation       |
| filename | string | A filename where a YAML representation of the interface should be placed |

This annotation, if present, will emit a
[YAML](https://en.wikipedia.org/wiki/YAML) representation of all interfaces that
were generated by the Grand Central views pass.  Equivalently, this is a
different serialization of the information contained in all `ViewAnnotation`s.

An example of this annotation is as follows:

``` json
{
  "class" : "sifive.enterprise.grandcentral.GrandCentralHierarchyFileAnnotation",
  "filename" : "directory/file.yaml"
}
```

The format of the produced YAML file is a one-to-one mapping of the
SystemVerilog interface to YAML.  Consider the following SystemVerilog interface
produced by GrandCentral:

``` systemverilog
interface Foo;
  // A 4-bit type
  logic [3:0] a;
  // A 2D vector of an 8-bit type
  logic [7:0] b [1:0][0:0];
  // A 1D vector of instances of Bar
  Bar bar[4]();
endinterface

interface Bar;
  logic c;
endinterface

interface Baz;
  logic d;
endinterface
```

This will produce the following YAML representation:

``` yaml
- name: Foo
  fields:
    - name: a
      description: A 4-bit type
      dimensions: [  ]
      width: 4
    - name: b
      description: A 2D vector of an 8-bit type
      dimensions: [ 1, 2 ]
      width: 8
  instances:
    - name: bar
      description: A 1D vector of instances of Bar
      dimensions: [ 4 ]
      interface:
        name: Bar
        fields:
          - name: c
            dimensions: [ ]
            width: 1
        instances: []
- name: Baz:
  fields:
    - name: d
      dimensions: [ ]
      width: 1
  instances: []
```

### Data Taps

Grand Central Taps are a tool for representing cross module references. They
enable users to "tap" into signal anywhere in the module hierarchy and treat
them as local, read-only signals.

`DataTaps` annotations are used to fill in the body of a FIRRTL external module
with cross-module references to other modules.  Each `DataTapKey` corresponds
to one output port on the `DataTapsAnnotation` external module.

#### ReferenceDataTapKey

| Property    | Type     | Description                                          |
| ----------- | -------- | ------------------                                   |
| class       | string   | `sifive.enterprise.grandcentral.ReferenceDataTapKey` |
| source      | string   | Reference target of the source signal.               |
| portName    | string   | Reference target of the data tap black box port      |

This key allows tapping a target in FIRRTL.

#### DataTapModuleSignalKey

| Property     | Type     | Description                                           |
| -----------  | -------- | ------------------                                    |
| class        | string   | sifive.enterprise.grandcentral.DataTapModuleSignalKey |
| module       | string   | ExtModule name of the target black box                |
| internalPath | string   | The path within the module                            |
| portName     | string   | Reference target of the data tap black box port       |

This key allows tapping a point by name in a blackbox.

#### LiteralDataTapKey

| Property    | Type     | Description                                      |
| ----------- | -------- | ------------------                               |
| class       | string   | sifive.enterprise.grandcentral.LiteralDataTapKey |
| literal     | string   | FIRRTL constant literal                          |
| portName    | string   | Reference target of the data tap black box port  |

This key allows the creation of a FIRRTL literal.

#### DataTapsAnnotation

| Property    | Type     | Description                                                       |
| ----------- | -------- | ------------------                                                |
| class       | string   | sifive.enterprise.grandcentral.DataTapsAnnotation                 |
| blackbox    | string   | ExtModule name of the black box with ports referenced by the keys |
| keys        | array    | List of DataTapKeys                                               |

The `DataTapsAnnotation` is a collection of all the data taps in a circuit.
This will cause a data tap module to be emitted.  The `DataTapsAnnotation`
implies `DontTouchAnnotation` on any `ReferenceDataTapKey.source` target.

Example:

```json
{
  "class": "sifive.enterprise.grandcentral.DataTapsAnnotation",
  "blackBox": "~GCTDataTap|DataTap",
  "keys": [
    {
      "class": "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      "source": "~GCTDataTap|GCTDataTap>r",
      "portName": "~GCTDataTap|DataTap>_0"
    },
    {
      "class":"sifive.enterprise.grandcentral.DataTapModuleSignalKey",
      "module":"~GCTDataTap|BlackBox",
      "internalPath":"baz.qux",
      "portName":"~GCTDataTap|DataTap>_1"
    },
    {
      "class":"sifive.enterprise.grandcentral.LiteralDataTapKey",
      "literal":"UInt<16>(\"h2a\")",
      "portName":"~GCTDataTap|DataTap>_3"
    }
  ]
}
```

### Memory Taps

Memory taps are a special version of data taps which are used for targeting the
FIRRTL memory vectors.

#### MemTapAnnotation

| Property    | Type             | Description                                                            |
| ----------- | --------         | ------------------                                                     |
| class       | string           | sifive.enterprise.grandcentral.MemTapAnnotation                        |
| taps        | array of strings | An array of components corresponding to the elements of the tap vector |
| source      | string           | Reference target to a FIRRTL memory element                            |

`MemoryTapAnnotation` is used to create a data tap to a FIRRTL memory. The
contents of the MemTap module are the cross-module references to each row of
the tapped memory. Attaching this annotation to memories with aggregate data
types is not supported.

Example:
```json
{
  "class": "sifive.enterprise.grandcentral.MemTapAnnotation",
  "taps":[
    "GCTMemTap.MemTap.mem[0]",
    "GCTMemTap.MemTap.mem[1]"
  ],
  "source":"~GCTMemTap|GCTMemTap>mem"
}
```

## Attributes in SV

Some annotations transform into attributes consumed by non-FIRRTL passes.  This
section describes well-defined attributes used by HW/SV passes.


### firrtl.moduleHierarchyFile

Used by HWExportModuleHierarchy.  Signifies a root from which to dump the module
hierarchy as a json file. This attribute is a list of files to output to, and
has type `ArrayAttr<OutputFileAttr>`.

The exported JSON file encodes a recursive tree of module instances as JSON
objects, with each object containing the following members:

- `instance_name` - A string describing the name of the instance. Note that the
  root module will have its `instance_name` set to the module's name.
- `module_name` - A string describing the name of the module.
- `instances` - An array of objects, where each object is a direct instance
  within the current module.

### firrtl.extract.assert

Used by SVExtractTestCode.  Specifies the output directory for extracted
modules. This attribute has type `OutputFileAttr`.

### firrtl.extract.assume

Used by SVExtractTestCode.  Specifies the output directory for extracted
modules. This attribute has type `OutputFileAttr`.

### firrtl.extract.cover

Used by SVExtractTestCode.  Specifies the output directory for extracted
modules. This attribute has type `OutputFileAttr`.

### firrtl.extract.assert.bindfile

Used by SVExtractTestCode.  Specifies the output file for extracted
modules' bind file. This attribute has type `OutputFileAttr`.

### firrtl.extract.assume.bindfile

Used by SVExtractTestCode.  Specifies the output file for extracted
modules' bind file. This attribute has type `OutputFileAttr`.

### firrtl.extract.cover.bindfile

Used by SVExtractTestCode.  Specifies the output file for extracted
modules' bind file. This attribute has type `OutputFileAttr`.

### firrtl.extract.[cover|assume|assert].extra

Used by SVExtractTestCode.  Indicates a module whose instances should be
extracted from the circuit in the indicated extraction type.


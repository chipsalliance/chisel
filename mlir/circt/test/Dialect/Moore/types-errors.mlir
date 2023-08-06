// RUN: circt-opt --verify-diagnostics --split-input-file %s

// expected-error @+1 {{ambiguous packing; wrap `named` in `packed<...>` or `unpacked<...>` to disambiguate}}
func.func @Foo(%arg0: !moore.named<"foo", bit, loc(unknown)>) { return }

// -----
// expected-error @+1 {{ambiguous packing; wrap `ref` in `packed<...>` or `unpacked<...>` to disambiguate}}
func.func @Foo(%arg0: !moore.ref<bit, loc(unknown)>) { return }

// -----
// expected-error @+1 {{ambiguous packing; wrap `unsized` in `packed<...>` or `unpacked<...>` to disambiguate}}
func.func @Foo(%arg0: !moore.unsized<bit>) { return }

// -----
// expected-error @+1 {{ambiguous packing; wrap `range` in `packed<...>` or `unpacked<...>` to disambiguate}}
func.func @Foo(%arg0: !moore.range<bit, 3:0>) { return }

// -----
// expected-error @+1 {{ambiguous packing; wrap `struct` in `packed<...>` or `unpacked<...>` to disambiguate}}
func.func @Foo(%arg0: !moore.struct<{}, loc(unknown)>) { return }

// -----
// expected-error @+1 {{unpacked struct cannot have a sign}}
func.func @Foo(%arg0: !moore.unpacked<struct<unsigned, {}, loc(unknown)>>) { return }
func.func @Bar(%arg0: !moore.packed<struct<unsigned, {}, loc(unknown)>>) { return }

// -----
// expected-error @+1 {{unpacked type '!moore.string' where only packed types are allowed}}
func.func @Foo(%arg0: !moore.packed<struct<{a: string}, loc(unknown)>>) { return }

// -----
// expected-error @+1 {{unpacked type '!moore.string' where only packed types are allowed}}
func.func @Foo(%arg0: !moore.packed<unsized<string>>) { return }

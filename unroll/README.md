# Unroll Fork

`unroll/plugin` contains a locally vendored fork of the upstream `com-lihaoyi/unroll`
compiler plugin.

Upstream project:
- https://github.com/com-lihaoyi/unroll

Licensing:
- The code under `unroll/plugin` retains the upstream MIT license.
- See `unroll/LICENSE` for the license text.
- Modifications to vendored code will retain the MIT license so they can be upstreamed as necessary.

Local use in this repository:
- This fork is built by `unroll/package.mill` as the local `unroll-plugin` compiler plugin.
- The published `unroll-annotation` dependency is still consumed from Maven.

Provenance:
- This subtree was imported from the upstream unroll project and may contain local modifications for Chisel's build and integration needs.
- This subtree was imported from commit c9fd7e1ce9a92fe4ab15cd12fb61facc3aed8f2c (tag 0.3.0).

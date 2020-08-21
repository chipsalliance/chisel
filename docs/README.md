# chisel3/docs/README.md

This directory contains documentation on the code within this repository.
Documents can either be written directly in markdown, or
use embedded [mdoc](https://scalameta.org/mdoc/)
which compiles against the `chisel3` (and dependencies) codebase
as part of the PR CI checks,
forcing the documentation to remain current with the codebase.
The `src` folder contains the source from which these are generated.

Previous Wiki documentation, now hosted by the website, is contained in the `src/wiki-deprecated` directory.
We are in the process of converting this documentation into the four categories as described in
[Divio's documentation system](https://documentation.divio.com/).

The four documentation types are:
 1. Reference (source code scaladoc)
 1. Explanation (`src/explanations`)
 1. How-To Guides (`src/cookbooks`)
 1. Tutorials (currently not located here)

Our documentation strategy for this repository is as follows:
 * Any new public API requires reference documentation.
 * Any new user-facing feature requires explanation documentation.
 * Any bugfixes, corner-cases, or answers to commonly asked questions requires a how-to guide.
 * For now, tutorials are kept in a separate repository. We are working hosting them here.
 * Old documentation is contained in the `src/wiki-deprecated` directory and is being incrementally converted to these
 categories.

To build the documentation, run `docs/mdoc` from SBT in the root directory. The generated documents
will appear in the `docs/generated` folder. To iterate on the documentation, you can run `docs/mdoc --watch`. For
more `mdoc` instructions you can visit their [website](https://scalameta.org/mdoc/).

This documentation is hosted on the Chisel [website](https://www.chisel-lang.org).

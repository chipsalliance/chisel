---
layout: docs
title:  "Naming Cookbook"
section: "chisel3"
---


### I still have _T signals, can this be fixed?

First check - is the compiler plugin properly enabled? If so, these signals could be intermediate values which are
consumed by either assertions or when predicates. In these cases, the compiler plugin often can't find a good prefix
for the generated intermediate signals. We recommend you manually insert calls to `prefix` to fix these cases. We did
this to RocketChip and saw huge benefits!

### I still see _GEN signals, can this be fixed?

`_GEN` signals are usually generated from the FIRRTL compiler, rather than the Chisel library. We are working on
renaming these signals with more context-dependent names, but it is a work in progress. Thanks for caring!

### My module names are super unstable - I change one thing and Queue_1 becomes Queue_42. Help!

This is the infamous `Queue` instability problem. In general, these cases are best solved at the source - the module
itself! If you overwrite `desiredName` to include parameter information (see the
[explanation](../explanations/naming.md) for more info), then this can avoid this problem permanantly. We've done this
with some Chisel utilities with great results!

### I want to add some hardware or assertions, but each time I do all the signal names get bumped!

This is the classic "ECO" problem, and we demonstrate a solution in [explanation](../explanations/naming.md). In short,
we recommend wrapping all additional logic in a prefix scope, which enables a unique namespace. This should prevent
name collisions, which are what triggers all those annoying signal name bumps!

### I want to force a signal (or instance) name to something, how do I do that?

Use the `.suggestName` method, which is on all classes which subtype 'Data'.

### All this prefixing is annoying, how do I fix it?

You can use the `noPrefix { ... }` to strip the prefix from all signals generated in that scope.

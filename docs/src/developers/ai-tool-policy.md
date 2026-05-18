---
layout: docs
title:  "AI Tool policy"
section: "chisel3"
---

# Chisel AI Tool Use Policy

## Summary

Contributors may use AI tools to assist with their work, but must:

- **Keep a human in the loop** - All AI-generated content must be reviewed and understood by the contributor before submission
- **Take full accountability** - The contributor is the author and is responsible for their contributions
- **Be transparent** - Label contributions containing AI-generated content with a message trailer: `Assisted-by: <tool>:<model>`, e.g., `Assisted-by: Claude Code:claude-sonnet-4-6`.  Include this trailer in commit messages, Pull Requests, or wherever authorship is normally indicated, regardless of the scope of the contribution.
- **Ensure quality** - Contributions should be worth more to the project than the time required to review them

## What This Means

**Allowed:**
- Using AI tools to generate code that you review, understand, and can explain
- Using AI for documentation that you verify for correctness
- Using AI to help debug or optimize code you understand

**Not Allowed:**
- Submitting AI-generated code without thorough human review
- Using automated agents that take action without human approval (e.g., GitHub `@claude` agent)
- Using AI tools to fix "good first issue" labeled issues (these are learning opportunities for newcomers)
- Passing maintainer feedback to an LLM without understanding and addressing it yourself

## Legal Requirements

Contributors using AI tools must still ensure they have the legal right to contribute code under the Apache-2.0 license.
Using AI to regenerate copyrighted material does not remove the copyright.

## References

This policy is closely related to and derived from the [CIRCT AI Tool Use policy](https://circt.llvm.org/docs/AIToolPolicy/) which is derived from the [LLVM AI Tool Use Policy](https://llvm.org/docs/AIToolPolicy.html).

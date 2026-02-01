## Compilation

- This repository uses Mill for compilation via the `./mill` bootstrap script
- You will need to load new enough Java with `module load oracle/graalvm/jdk/17.0.7` before running any Mill command
- To build the project, run `./mill chisel[_].test.compile`
- This project is cross-compiled for Scala 2.13 and Scala 3, we need to check that both work.
- This project uses 2.13 and 3 as the cross versions, e.g. `./mill chisel[2.13].compile` compiles for just 2.13.

## Testing

- The full suite of tests takes a long time so please do not run it.
- Instead, please try to identify one or more tests in `src/test` that checks the feature or behavior you are touching. For example `./mill chisel[_].test.testOnly chiselTests.LTLSpec`
- Sometimes tests may be in `src/test/scala-2` which means they only currently are compiled for Scala 2. If you find relevant tests in `src/test/scala-2`, please move them to `src/main/scala` and fix any issues for Scala 3 in a separate commit.

## Coding Guidelines

- Please run formatting before committing with `./mill __.reformat`
- Do not commit whitespace changes.
- If you notice that there are whitespace changes that should be made, e.g., removing trailing newlines, factor these out into a separate NFC commit.
- All changes should include a test of the change.
- Where possible, add new tests to existing test files.

## Commit Messages

- The commit message title should be terse and aim for 50 characters.
- Use "tags" for commit message titles, e.g., `[scala3]` or `[nfc]`.
- The commit message body should be line-wrapped to 74 characters.
- For any commit message you author, include a `Co-authored-by` line with your name and email.

## General Guidance

- If you get stuck, that's fine!  You're working hard and are appreciated.
- When you get stuck, ask for directions or clarification.
- If you think you are getting stuck in a loop, e.g., you are repeatedly applying and then removing a change, stop and ask for help.
- We're a team and we're working together.
- You may create commits.
- NEVER push code to a repository.

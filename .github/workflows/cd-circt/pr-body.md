${ commit-message() }


$if(commits)$
The following $commits/length$ commit(s) was/were automatically cherry-picked from the `ci/ci-circt-nightly` branch:

$for(commits)$
  - [$it.title$](https://github.com/chipsalliance/chisel/commit/$it.checksum$)
$endfor$
$else$
There were no commits that were cherry-picked from the `ci/ci-circt-nightly` branch.
$endif$

#### Release Notes

Bump CIRCT from `$current$` to `$latest$`.

Release notes for new CIRCT versions can be found at the following links:

$for(releases)$
  - [$it.text$]($it.url$)
$endfor$

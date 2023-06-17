{ stdenv, fetchFromGitHub, cmake, ninja }:
stdenv.mkDerivation rec {
  pname = "espresso";
  version = "2.4";
  nativeBuildInputs = [ cmake ninja ];
  src = fetchFromGitHub {
    owner = "chipsalliance";
    repo = "espresso";
    rev = "v${version}";
    sha256 = "sha256-z5By57VbmIt4sgRgvECnLbZklnDDWUA6fyvWVyXUzsI=";
  };
}


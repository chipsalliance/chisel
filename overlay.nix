final: prev:
{
  mill = prev.mill.override { jre = final.openjdk20; };
}

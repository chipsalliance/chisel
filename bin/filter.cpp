#include <stdio.h>

int main (int argc, char* argv[]) {
  int c;
  for (;;) {
    c = getchar();
    if (c == EOF) break;
    char nc = (c == '$' || c == '#') ? '_' : c;
    putchar(nc);
  } 
}

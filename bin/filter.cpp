#include <stdio.h>

int main (int argc, char* argv[]) {
  int c;
  for (;;) {
    c = getchar();
    if (c == EOF) break;
    if (c == '#')
      putchar('_');
    else if (c == '$') {
      putchar(':'); putchar(':');
    } else
      putchar(c);
  } 
}

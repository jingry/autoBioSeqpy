use strict;
use warnings;

while (<>) {
     chomp;
     if(/>(.+)/) {
         print "$1\n";
       }
}
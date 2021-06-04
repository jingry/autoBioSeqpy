use strict;
use warnings;

while (<>) {
     chomp;
     if(/>(.+?) /) {
         print "\n>$1\n";
       } else {
         print "$_";
     }
}
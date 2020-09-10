#!/bin/bash
times=(1 10 100 250 500 1000)
testdir=tests
for n in ${times[@]};
do
	echo "Compiling test $n"
	nvcc -O3 wa1-task3.cu -DN_ELEMS=$n -o $testdir/$n
	out=$($testdir/$n)
	echo -e "TEST $n\n$out"
done
	

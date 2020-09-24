##Feedback, assignment 2

* Student: Christian Påbøl Jacobsen
* Points: 5.5/10

### Overall

### Task 1: flat parallel prime sieve in Futhark - 0/3 pts
* Not attempted.

### Task 2: coalesced memory access in CUDA - 1.5/2 pts
* Your new formula for computing the local ID is exactly correct!

* Good explanation of why the change achieves coalesced access.

* Your benchmark results are fine, but I am lacking a little description of your
  benchmarking strategy. In particular, which GPUs, input sizes, and block
  sizes were used to produce these results?


* __If you want to resubmit, please include:__
  - short description of your benchmarking strategy (include the points I asked
    for above + anything else of relevance, if any at all)

### Task 3: warp-level inclusive scan in CUDA - 2/2 pts

* Your re-implementation of `scanIncWarp` is also exactly correct! No complaints
  here.

* I am missing some description of the code, or at least some code comments. Did
  you for example use the lecture notes as inspiration?

* Nice table in section 2.1 ;) although I am still missing information on
  important such as the ones requested in the resubmission for task 2, but I'll
  ignore it for now. Just remember that in the final exam project, this type of
  information is crucial to the validity/comparability of benchmark results.

### Task 4: finding the bug in `scanIncblock` - 1/1 pts

* Excellent analysis and description of the race condition!

* You also present a good and effective fix for the race condition. Except for
  the "cheating" solution using `res` (which only works because `scanIncWarp`
  does not return void), this is the most optimal solution.

* You present an alternative solution which you call `Another more robust fix`.
  How is this more robust? Answer: it is *not* more robust than your solution, so
  long as you do not compile to a CUDA-compatible architecture with more than 32
  warps per thread block (which you likely won't).

* You forgot to actually assert whether your changes actually fixed the race
  condition (ie. by running a validation test with `BLOCK_SIZE=1024`) ;)

### Task 5: flat sparse matrix-vector multiplication in CUDA - 1/2 pts

* Very good implementation! Your program seems correct and efficient. Good
  comments in the snippet in the report, but please also include these in the
  actual code file (I prefer looking at source code he)


* I am not quite able to reproduce your benchmark results using `(num_rows,
  vct_size, block_size) == (11033, 2076, 256)` on gpu04. I get the same CPU
  result, but `756` microseconds for the GPU implementation. Are you sure this
  is the correct result?

* Remember to validate your program!! Or rather, to report on the results of your
  validation tests, hehe.


* __If you want to resubmit, please include:__
  - explicit validation testing (ie. you already did it but please report on it)
  - please re-run your benchmark; whether you get the same result or one more
    similar to mine, please comment on it in the report, and describe how you
    executed your benchmarks (similarly to the resubmission comment for task 2).

### OPTIONAL task 6: parallel partition in CUDA
* Not attempted.

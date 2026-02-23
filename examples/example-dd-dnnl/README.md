# example-nd-mkl

An example that shows how to use BLAS-like Extensions for [Neanderthal](https://neanderthal.uncomplicate.org/) 
in conjunction with [Deep Diamond](https://github.com/uncomplicate/deep-diamond)

## Try out batched GEMM

    $ lein repl

    nREPL server started on port 35965 on host 127.0.0.1 - nrepl://127.0.0.1:35965
    REPL-y 0.5.1, nREPL 1.0.0
    Clojure 1.12.3
    ...
    
    user=> (load-file "src/batch_gemm.clj")
    INFO: Loading :mkl backend. It may take a few seconds. Please stand by.
    INFO: MKL backend loaded.
    Verification Batch 0 (Should be true): true
    Verification Batch 1 (Should be true): true
    Result 0:
    #RealGEMatrix[float, mxn:2x3, layout:column]
    ▥       ↓       ↓       ↓       ┓    
    →       1.00    3.00    5.00         
    →       2.00    4.00    6.00         
    ┗                               ┛
    
    Result 1:
    #RealGEMatrix[float, mxn:2x3, layout:column]
    ▥       ↓       ↓       ↓       ┓    
    →       7.00    9.00   11.00         
    →       8.00   10.00   12.00         
    ┗                               ┛
    
    user=> (load-file "src/batch_axpy.clj")
    Verification Batch 0 (Should be true): true
    Verification Batch 1 (Should be true): true
    Result Y0:
    #RealBlockVector[float, n:4, stride:1]
    [  12.00   24.00   36.00   48.00 ]
    Result Y1:
    #RealBlockVector[float, n:4, stride:1]
    [  60.00   72.00   84.00   96.00 ]


## License

Copyright © 2026 Kamil Toman

This program and the accompanying materials are made available under the
terms of the Eclipse Public License 2.0 which is available at
http://www.eclipse.org/legal/epl-2.0.

This Source Code may also be made available under the following Secondary
Licenses when the conditions for such availability set forth in the Eclipse
Public License, v. 2.0 are satisfied: GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or (at your
option) any later version, with the GNU Classpath Exception which is available
at https://www.gnu.org/software/classpath/license.html.

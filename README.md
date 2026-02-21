# Neanderthal-BLAS-like

BLAS-like Extensions for [Neanderthal](https://neanderthal.uncomplicate.org/)

## Motivation

Neanderthal currently lacks some optional functions for efficient batching of BLAS operations that are typically present in its underlying libraries. This project provides these extensions, which can be useful in certain scenarios that heavily use batched matrix computations.

Added functions:
- mm-batch-strided!
- axpy-batch-strided!

## Getting started

Add the necessary dependency to your project:

```$clojure
    [neanderthal-blas-like "0.1.0"]
```

## Running Tests

Use the development profile so that the native libraries are available when
running the Midje test suite:

```bash
lein with-profile +dev midje
```

## Usage

To see how to use these BLAS-like extensions in practice, please check out the `examples` directory. You will find examples demonstrating the usage with:

*   **Pure Neanderthal:** Check the `example-nd-mkl` directory for how to integrate these extensions with standard Neanderthal operations using the MKL backend.
*   **Deep Diamond:** Check the `example-dd-dnnl` directory to see how this library can be utilized in conjunction with Deep Diamond.

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

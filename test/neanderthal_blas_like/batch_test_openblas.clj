;;   Copyright (c) Kamil Toman. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 2.0 (https://opensource.org/licenses/EPL-2.0) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns neanderthal-blas-like.batch-test-openblas
  (:require [midje.sweet :refer [facts]]
            [uncomplicate.neanderthal.native :refer [native-float native-double]]
            [uncomplicate.neanderthal.internal.cpp.openblas.factory :refer [openblas-float openblas-double]]
            [neanderthal-blas-like.be.openblas :refer :all]
            [neanderthal-blas-like.batch-test-common-ge :as ge]
            [neanderthal-blas-like.batch-test-common-vctr :as v]))

(facts "OpenBLAS Batched BLAS (Fallback) GE float"
       (ge/run-all-tests native-float openblas-float))

(facts "OpenBLAS Batched BLAS (Fallback) GE double"
       (ge/run-all-tests native-double openblas-double))

(facts "OpenBLAS Batched BLAS (Fallback) vector float"
       (v/run-all-tests native-float openblas-float))

(facts "OpenBLAS Batched BLAS (Fallback) vector double"
       (v/run-all-tests native-double openblas-double))

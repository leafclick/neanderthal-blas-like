;;   Copyright (c) Kamil Toman. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 2.0 (https://opensource.org/licenses/EPL-2.0) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns neanderthal-blas-like.batch-test-mkl
  (:refer-clojure :exclude [abs])
  (:require [midje.sweet :refer [facts]]
            [uncomplicate.neanderthal.internal.api :refer :all]
            [uncomplicate.neanderthal.internal.cpp.mkl.factory
             :refer [mkl-float mkl-double]]
            [neanderthal-blas-like.batch-test-common-ge :as ge]
            [neanderthal-blas-like.batch-test-common-vctr :as v]
            [neanderthal-blas-like.be.mkl :refer :all]))

(facts "MKL Batched BLAS GE float"
       (ge/run-all-tests mkl-float mkl-float))

(facts "MKL Batched BLAS GE double"
       (ge/run-all-tests mkl-double mkl-double))

(facts "MKL Batched BLAS vector float"
       (v/run-all-tests mkl-float mkl-float))

(facts "MKL Batched BLAS vector double"
       (v/run-all-tests mkl-double mkl-double))
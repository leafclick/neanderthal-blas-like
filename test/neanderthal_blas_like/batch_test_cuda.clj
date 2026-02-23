;;   Copyright (c) Kamil Toman. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 2.0 (https://opensource.org/licenses/EPL-2.0) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns neanderthal-blas-like.batch-test-cuda
  (:refer-clojure :exclude [abs])
  (:require [midje.sweet :refer [facts fact roughly just =>]]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal.internal.api :refer :all]
            [uncomplicate.clojurecuda.core :refer [with-default default-stream]]
            [uncomplicate.neanderthal.cuda :refer [with-engine *cuda-factory*
                                                   cuda-float cuda-double]]
            [neanderthal-blas-like.batch-test-common-ge :as ge]
            [neanderthal-blas-like.batch-test-common-vctr :as v]
            [neanderthal-blas-like.be.cuda :refer :all]))

(with-default

  (facts "CUDA Batched BLAS GE float"
         (with-engine cuda-float default-stream
           (ge/run-all-tests (native-factory *cuda-factory*) *cuda-factory*)))
  (facts "CUDA Batched BLAS GE float"
         (with-engine cuda-float default-stream
           (ge/run-all-tests (native-factory *cuda-factory*) *cuda-factory*)))

  (facts "CUDA Batched BLAS vector double"
         (with-engine cuda-double default-stream
           (v/run-all-tests (native-factory *cuda-factory*) *cuda-factory*)))
  (facts "CUDA Batched BLAS vector double"
         (with-engine cuda-double default-stream
           (v/run-all-tests (native-factory *cuda-factory*) *cuda-factory*))))
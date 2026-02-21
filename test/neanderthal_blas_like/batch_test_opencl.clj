;;   Copyright (c) Kamil Toman. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 2.0 (https://opensource.org/licenses/EPL-2.0) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns neanderthal-blas-like.batch-test-opencl
  (:require [midje.sweet :refer [facts fact roughly just =>]]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal.internal.api :refer :all]
            [uncomplicate.clojurecl.core :refer [with-default-1 *command-queue*]]
            [uncomplicate.neanderthal.opencl :refer [with-engine *opencl-factory*
                                                     opencl-float opencl-double]]
            [neanderthal-blas-like.batch-test-common-ge :as ge]
            [neanderthal-blas-like.batch-test-common-vctr :as v]
            [neanderthal-blas-like.be.opencl :refer :all]))

(with-default-1
  (facts "OpenCL Batched BLAS GE float"
         (with-engine opencl-float *command-queue*
                      (ge/run-all-tests (native-factory *opencl-factory*) *opencl-factory*)))
  (facts "OpenCL Batched BLAS GE double"
         (with-engine opencl-double *command-queue*
                      (ge/run-all-tests (native-factory *opencl-factory*) *opencl-factory*)))
  (facts "OpenCL Batched BLAS vector float"
         (with-engine opencl-float *command-queue*
                      (v/run-all-tests (native-factory *opencl-factory*) *opencl-factory*)))
  (facts "OpenCL Batched BLAS vector double"
         (with-engine opencl-double *command-queue*
                      (v/run-all-tests (native-factory *opencl-factory*) *opencl-factory*)))
  )

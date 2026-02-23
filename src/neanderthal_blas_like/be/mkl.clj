;;   Copyright (c) Kamil Toman. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 2.0 (https://opensource.org/licenses/EPL-2.0) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns neanderthal-blas-like.be.mkl
  (:refer-clojure :exclude [abs])
  (:require [neanderthal-blas-like.internal.api :refer :all]
            [uncomplicate.neanderthal.core :refer [mrows ncols dim]]
            [uncomplicate.neanderthal.internal.cpp.mkl.factory]
            [uncomplicate.neanderthal.internal.constants :refer :all]
            [uncomplicate.neanderthal.internal.api :refer [Navigable navigator]]
            [uncomplicate.neanderthal.block :refer [stride]]
            [uncomplicate.neanderthal.internal.cpp
             [common :refer :all]
             [structures :refer :all]
             [blas :refer :all]
             [lapack :refer :all]
             [factory :refer :all :exclude [cast-stream]]])
  (:import [uncomplicate.neanderthal.internal.api LayoutNavigator]
           [uncomplicate.neanderthal.internal.cpp.mkl.factory DoubleVectorEngine FloatGEEngine DoubleGEEngine FloatVectorEngine]
           [org.bytedeco.mkl.global mkl_rt]))

(defmacro mkl-real-ge-batched-blas* [name t ptr cast blas]
  `(extend-type ~name
     BatchedBlas
     (gemm-batch-strided [_# alpha# a# stride-a# b# stride-b# beta# c# stride-c# batch-size#]
       (. ~blas ~(cblas t 'gemm_batch_strided)
          (.layout ^LayoutNavigator (navigator c#))
          (if (= (navigator c#) (navigator a#)) ~(:no-trans blas-transpose) ~(:trans blas-transpose))
          (if (= (navigator c#) (navigator b#)) ~(:no-trans blas-transpose) ~(:trans blas-transpose))
          (int (mrows a#)) (int (ncols b#)) (int (ncols a#))
          (~cast alpha#) (~ptr a#) (int (stride a#)) (int stride-a#)
          (~ptr b#) (int (stride b#)) (int stride-b#)
          (~cast beta#) (~ptr c#) (int (stride c#)) (int stride-c#)
          (int batch-size#))
       c#)))

(defmacro mkl-real-batched-blas* [name t ptr cast blas]
  `(extend-type ~name
     BatchedBlas
     (axpy-batch-strided [_# n# alpha# x# stride-x# y# stride-y# batch-size#]
                         (. ~blas ~(cblas t 'axpy_batch_strided)
                            (int n#) (~cast alpha#)
                            (~ptr x#) (int (stride x#)) (int stride-x#)
                            (~ptr y#) (int (stride y#)) (int stride-y#)
                            (int batch-size#))
                         y#)))

(mkl-real-ge-batched-blas* FloatGEEngine "s" float-ptr float mkl_rt)
(mkl-real-ge-batched-blas* DoubleGEEngine "d" double-ptr double mkl_rt)
(mkl-real-batched-blas* FloatVectorEngine "s" float-ptr float mkl_rt)
(mkl-real-batched-blas* DoubleVectorEngine "d" double-ptr double mkl_rt)


;;   Copyright (c) Kamil Toman. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 2.0 (https://opensource.org/licenses/EPL-2.0) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns neanderthal-blas-like.be.openblas
  (:refer-clojure :exclude [abs])
  (:require [neanderthal-blas-like.internal.api :refer [BatchedBlas gemm-batch-strided axpy-batch-strided]]
            [uncomplicate.neanderthal.core :refer [mrows ncols dim mm! axpy! subvector]]
            [uncomplicate.neanderthal.internal.api :as iapi]
            [uncomplicate.neanderthal.internal.cpp.openblas.factory])
  (:import [uncomplicate.neanderthal.internal.cpp.openblas.factory
            FloatGEEngine DoubleGEEngine FloatVectorEngine DoubleVectorEngine]
           [uncomplicate.neanderthal.internal.api GEMatrix Matrix Vector LayoutNavigator]))

(defn- strided-ge
  "Return a GE view into the same buffer as `a`, offset by `element-offset` elements,
   with dimensions m × n. Works by going through the internal submatrix mechanism."
  [^GEMatrix a ^long m ^long n ^long element-offset]
  (let [nav (iapi/navigator a)
        col-major? (.isColumnMajor ^LayoutNavigator nav)
        col-offset (if col-major? (quot element-offset m) (rem element-offset n))
        row-offset (if col-major? (rem element-offset m) (quot element-offset n))]
    (.submatrix ^Matrix a row-offset col-offset m n)))

(defmacro openblas-real-ge-batched-blas* [name]
  `(extend-type ~name
     BatchedBlas
     (gemm-batch-strided [_# alpha# a# stride-a# b# stride-b# beta# c# stride-c# batch-size#]
       (let [m# (long (mrows a#))
             ka# (long (ncols a#))
             kb# (long (mrows b#))
             n# (long (ncols b#))
             stride-a# (long stride-a#)
             stride-b# (long stride-b#)
             stride-c# (long stride-c#)
             batch-size# (long batch-size#)]
         (dotimes [i# batch-size#]
           (let [ai# (strided-ge a# m# ka# (* (long i#) stride-a#))
                 bi# (strided-ge b# kb# n# (* (long i#) stride-b#))
                 ci# (strided-ge c# m# n# (* (long i#) stride-c#))]
             (mm! alpha# ai# bi# beta# ci#))))
       c#)))

(defmacro openblas-real-batched-blas* [name]
  `(extend-type ~name
     BatchedBlas
     (axpy-batch-strided [_# n# alpha# x# stride-x# y# stride-y# batch-size#]
                         (let [n# (long n#)
                               stride-x# (long stride-x#)
                               stride-y# (long stride-y#)
                               batch-size# (long batch-size#)]
                           (dotimes [i# batch-size#]
                             (let [xi# (.subvector ^Vector x# (* (long i#) stride-x#) n#)
                                   yi# (.subvector ^Vector y# (* (long i#) stride-y#) n#)]
                               (axpy! alpha# xi# yi#))))
                         y#)))

(openblas-real-ge-batched-blas* FloatGEEngine)
(openblas-real-ge-batched-blas* DoubleGEEngine)
(openblas-real-batched-blas* FloatVectorEngine)
(openblas-real-batched-blas* DoubleVectorEngine)

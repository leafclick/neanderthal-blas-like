;;   Copyright (c) Kamil Toman. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 2.0 (https://opensource.org/licenses/EPL-2.0) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns neanderthal-blas-like.be.cuda
  (:refer-clojure :exclude [abs])
  (:require [neanderthal-blas-like.internal.api :refer :all]
            [clojure.java.io :as io]
            [uncomplicate.commons.core :refer [with-release let-release Releaseable release]]
            [uncomplicate.neanderthal.core :refer [mrows ncols]]
            [uncomplicate.neanderthal.internal.api :refer [navigator data-accessor flow]]
            [uncomplicate.neanderthal.block :refer [stride offset]]
            [uncomplicate.neanderthal.internal.cpp.cuda.factory]
            [uncomplicate.neanderthal.internal.cpp.cuda.constants :refer :all]
            [uncomplicate.neanderthal.internal.cpp.cuda.structures :refer [cublas-error get-context]]
            [uncomplicate.commons.utils :refer [with-check]]
            [uncomplicate.clojure-cpp :as cpp]
            [uncomplicate.clojurecuda.core :as cuda]
            [uncomplicate.clojurecuda.info :refer [driver-version]]
            [uncomplicate.fluokitten.protocols :refer [extract]]
            [uncomplicate.neanderthal.internal.cpp.common :refer [float-ptr double-ptr]])
  (:import [uncomplicate.neanderthal.internal.cpp.cuda.factory
            FloatGEEngine DoubleGEEngine FloatVectorEngine DoubleVectorEngine HandleProvider]
           [uncomplicate.neanderthal.internal.api Block Matrix DataAccessor RealNativeVector
                                                  IntegerNativeVector RealNativeMatrix IntegerNativeMatrix
                                                  FullStorage Region Default GEMatrix CUMatrix UploMatrix
                                                  IntegerVector LayoutNavigator MatrixImplementation RealAccessor IntegerAccessor
                                                  CUVector CUMatrix CLVector CLMatrix Changeable]
           [org.bytedeco.cuda.cublas cublasContext]
           [org.bytedeco.cuda.global cublas]))

(def ^:private custom-cuda-source
  (slurp (io/resource "cuda/vector_blas_like.cu")))

(deftype CustomCudaModule [modl]
  Releaseable
  (release [_]
    (release modl)
    true))

(def ^:private context-modules (atom {}))

(defn- get-custom-module [ctx type]
  (let [ctx-ptr (extract ctx)
        modules @context-modules]
    (if-let [custom-modl (get-in modules [ctx-ptr type])]
      (.-modl ^CustomCudaModule custom-modl)
      (let [prog (cuda/compile! (cuda/program custom-cuda-source)
                                [(str "-DNUMBER=" type)
                                 (str "-DREAL=" type)
                                 (str "-DACCUMULATOR=" type)
                                 (if (= type "float") "-DCAST(fun)=fun##f" "-DCAST(fun)=fun")
                                 "-default-device"
                                 (format "-DCUDART_VERSION=%s" (driver-version))])
            modl (cuda/module prog)
            custom-modl (->CustomCudaModule modl)]
        (swap! context-modules assoc-in [ctx-ptr type] custom-modl)
        modl))))

(defmacro cuda-real-ge-batched-blas* [name t ptr cpp-ptr]
  `(extend-type ~name
     BatchedBlas
     (gemm-batch-strided [this# alpha# a# stride-a# b# stride-b# beta# c# stride-c# batch-size#]
       (if (< 0 (long (uncomplicate.neanderthal.core/dim a#)))
         (let [da# (uncomplicate.neanderthal.internal.api/data-accessor a#)
               nav-c# (navigator c#)
               stor-c# (uncomplicate.neanderthal.internal.navigation/full-storage c#)
               [x# y# trans-x# trans-y#]
               (if (.isColumnMajor ^LayoutNavigator nav-c#)
                 [a# b# (= nav-c# (navigator a#)) (= nav-c# (navigator b#))]
                 [b# a# (= nav-c# (navigator b#)) (= nav-c# (navigator a#))])]
           (with-release [alpha-p# (~cpp-ptr (.wrapPrim da# alpha#))
                          beta-p# (~cpp-ptr (.wrapPrim da# beta#))]
             (with-check cublas-error
                         (. cublas ~(symbol (str "cublas" t "gemmStridedBatched"))
                            ^cublasContext (.handle ^HandleProvider this#)
                            (int (if trans-x# ~(:no-trans cublas-trans) ~(:trans cublas-trans)))
                            (int (if trans-y# ~(:no-trans cublas-trans) ~(:trans cublas-trans)))
                            (int (.sd stor-c#)) (int (.fd stor-c#)) (int (ncols a#))
                            alpha-p# (~ptr x#) (int (stride x#)) (long stride-a#)
                            (~ptr y#) (int (stride y#)) (long stride-b#)
                            beta-p# (~ptr c#) (int (.ld stor-c#)) (long stride-c#)
                            (int batch-size#))
                         c#))))
       c#)))

(defmacro cuda-real-batched-blas* [name t ptr cpp-ptr kernel-type]
  `(extend-type ~name
     BatchedBlas
     (axpy-batch-strided [this# n# alpha# x# stride-x# y# stride-y# batch-size#]
                         (let [n# (long n#)
                               batch-size# (long batch-size#)
                               total# (* n# batch-size#)]
                           (when (< 0 total#)
                             (let [da# (data-accessor x#)
                                   ctx# (get-context da#)
                                   modl# (get-custom-module ctx# ~kernel-type)
                                   hstream# (flow da#)]
                               (with-release [kernel# (cuda/function modl# "axpy_batch_strided")]
                                 (cuda/launch! kernel# (cuda/grid-1d total# 256) hstream#
                                               (cuda/parameters (int n#)
                                                                (.castPrim da# alpha#)
                                                                (extract x#) (int (offset x#)) (int (stride x#)) (long stride-x#)
                                                                (extract y#) (int (offset y#)) (int (stride y#)) (long stride-y#)
                                                                (int batch-size#))))))
                           y#))))

(cuda-real-ge-batched-blas* FloatGEEngine "S" float-ptr cpp/float-ptr)
(cuda-real-ge-batched-blas* DoubleGEEngine "D" double-ptr cpp/double-ptr)
(cuda-real-batched-blas* FloatVectorEngine "S" float-ptr cpp/float-ptr "float")
(cuda-real-batched-blas* DoubleVectorEngine "D" double-ptr cpp/double-ptr "double")

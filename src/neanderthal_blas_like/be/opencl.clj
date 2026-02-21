;;   Copyright (c) Kamil Toman. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 2.0 (https://opensource.org/licenses/EPL-2.0) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns neanderthal-blas-like.be.opencl
  (:require [clojure.java.io :as io]
            [neanderthal-blas-like.internal.api :refer [BatchedBlas gemm-batch-strided axpy-batch-strided]]
            [uncomplicate.neanderthal.core :refer [ncols dim]]
            [uncomplicate.neanderthal.internal.api :refer [navigator]]
            [uncomplicate.neanderthal.internal.navigation :refer [full-storage]]
            [uncomplicate.neanderthal.block :refer [stride offset buffer]]
            [uncomplicate.fluokitten.protocols :refer [extract]]
            [uncomplicate.commons.utils :refer [with-check]]
            [uncomplicate.clojurecl.core :as cl]
            [uncomplicate.clojurecl.info :as cl-info]
            [uncomplicate.neanderthal.internal.device.clblast])
  (:import [uncomplicate.neanderthal.internal.device.clblast
            FloatGEEngine DoubleGEEngine FloatVectorEngine DoubleVectorEngine]
           [uncomplicate.neanderthal.internal.api LayoutNavigator]
           [org.jocl.blast CLBlast CLBlastLayout CLBlastStatusCode CLBlastTranspose]))


(defn ^:private clblast-error [^long err-code details]
  (let [err (CLBlastStatusCode/stringFor err-code)]
    (ex-info (format "CLBlast error: %s." err)
             {:name err :code err-code :type :clblast-error :details details})))

(def ^:private custom-opencl-source
  (slurp (io/resource "opencl/vector_blas_like.cl")))

(def ^:private context-modules (atom {}))

(defn- get-custom-program [queue t]
  (let [ctx (cl-info/queue-context queue)
        ctx-ptr (extract ctx)
        modules @context-modules]
    (if-let [prog (get-in modules [ctx-ptr t])]
      prog
      (let [dev (cl-info/queue-device queue)
            wgs (cl-info/max-work-group-size dev)
            apple? (clojure.string/includes? (cl-info/name-info (cl-info/platform dev)) "Apple")
            options (format "-DREAL=%s -DWGS=%d -DNATIVE(fun)=%s"
                            t wgs (if apple? "fun" "native_##fun"))
            prog (cl/build-program! (cl/program-with-source ctx [custom-opencl-source]) options nil)]
        (swap! context-modules assoc-in [ctx-ptr t] prog)
        prog))))

(defmacro ^:private opencl-real-ge-batched-blas* [name method]
  `(extend-type ~name
     BatchedBlas
     (gemm-batch-strided [this# alpha# a# stride-a# b# stride-b# beta# c# stride-c# batch-size#]
       (if (< 0 (long (dim a#)))
         (let [nav-c# (navigator c#)
               stor-c# (full-storage c#)
               [x# y# trans-x# trans-y#]
               (if (.isColumnMajor ^LayoutNavigator nav-c#)
                 [a# b# (= nav-c# (navigator a#)) (= nav-c# (navigator b#))]
                 [b# a# (= nav-c# (navigator b#)) (= nav-c# (navigator a#))])]
           (with-check clblast-error
                       (~method
                         CLBlastLayout/CLBlastLayoutColMajor
                         (if trans-x# CLBlastTranspose/CLBlastTransposeNo CLBlastTranspose/CLBlastTransposeYes)
                         (if trans-y# CLBlastTranspose/CLBlastTransposeNo CLBlastTranspose/CLBlastTransposeYes)
                         (long (.sd stor-c#)) (long (.fd stor-c#)) (long (ncols a#))
                         (double alpha#)
                         (extract (buffer x#)) (long (offset x#)) (long (stride x#)) (long stride-a#)
                         (extract (buffer y#)) (long (offset y#)) (long (stride y#)) (long stride-b#)
                         (double beta#)
                         (extract (buffer c#)) (long (offset c#)) (long (.ld stor-c#)) (long stride-c#)
                         (long batch-size#)
                         (extract (.queue this#)) nil)
                       c#))
         c#))))

(defmacro ^:private opencl-real-batched-blas* [name type-str]
  `(extend-type ~name
     BatchedBlas
     (axpy-batch-strided [this# n# alpha# x# stride-x# y# stride-y# batch-size#]
                         (let [n# (long n#)
                               batch-size# (long batch-size#)
                               total# (* n# batch-size#)]
                           (when (< 0 total#)
                             (let [queue# (.queue this#)
                                   prog# (get-custom-program queue# ~type-str)
                                   kernel# (cl/kernel prog# "axpy_batch_strided")]
                               (cl/set-args! kernel#
                                             (int-array [(int n#)])
                                             (if (= ~type-str "double") (double-array [(double alpha#)]) (float-array [(float alpha#)]))
                                             (buffer x#) (int-array [(int (offset x#))]) (int-array [(int (stride x#))]) (long-array [(long stride-x#)])
                                             (buffer y#) (int-array [(int (offset y#))]) (int-array [(int (stride y#))]) (long-array [(long stride-y#)])
                                             (int-array [(int batch-size#)]))
                               (cl/enq-kernel! queue# kernel# (cl/work-size [total#]))))
                           y#))))

(opencl-real-ge-batched-blas* FloatGEEngine CLBlast/CLBlastSgemmStridedBatched)
(opencl-real-ge-batched-blas* DoubleGEEngine CLBlast/CLBlastDgemmStridedBatched)
(opencl-real-batched-blas* FloatVectorEngine "float")
(opencl-real-batched-blas* DoubleVectorEngine "double")

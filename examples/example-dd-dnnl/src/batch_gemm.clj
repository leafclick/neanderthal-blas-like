(ns batch-gemm
  (:require [uncomplicate.neanderthal.core :refer [view-vctr transfer! view-ge subvector]]
            [uncomplicate.neanderthal.internal.api :as api]
            [uncomplicate.neanderthal.native :refer [fv fge]]
            [uncomplicate.diamond.tensor :as tz :refer [view-tz offset!]]
            [uncomplicate.diamond.internal.dnnl.factory :refer [dnnl-factory]]
            [uncomplicate.commons.core :refer [Info info with-release]]
            [neanderthal-blas-like.batch :refer [mm-batch-strided! batch-ge]]
            [neanderthal-blas-like.be.mkl]))

(defn batch-stride [tz] (first (tz/layout tz)))
(defn view-batch [tz idx]
      (let [sub (into [1] (rest (tz/shape tz)))]
           (-> (doto (tz/view-tz tz sub)
                     (tz/offset! (* idx (batch-stride tz))))
               (view-vctr))))

(let [fact (dnnl-factory)
      batch-size 2]
     (with-release [tensor-a (tz/tensor fact [batch-size 2 3] :float :ncw)
                    tensor-b (tz/tensor fact [batch-size 3 3] :float :ncw)
                    tensor-c (tz/tensor fact [batch-size 2 3] :float :ncw)]

                   ;; 1. Setup A: Batch of 2 (2x3 matrices)
                   (let [a0-data (fv [1.0 2.0 3.0 4.0 5.0 6.0])
                         a1-data (fv [7.0 8.0 9.0 10.0 11.0 12.0])]
                        (transfer! a0-data (view-batch tensor-a 0))
                        (transfer! a1-data (view-batch tensor-a 1))

                        ;; 2. Setup B: Batch of 2 (3x3 Identity matrices)
                        (let [identity-3x3 (fv [1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0])]
                             (transfer! identity-3x3 (view-batch tensor-b 0))
                             (transfer! identity-3x3 (view-batch tensor-b 1))

                             ;; 3. Batched Multiply: C = A * B
                             (mm-batch-strided! 1.0
                                                (view-ge (view-batch tensor-a 0) 2 3) (batch-stride tensor-a) ;; stride-a: 2x3
                                                (view-ge (view-batch tensor-b 0) 3 3) (batch-stride tensor-b) ;; stride-b: 3x3
                                                0.0
                                                (view-ge (view-batch tensor-c 0) 2 3) (batch-stride tensor-c) ;; stride-c: 2x3
                                                batch-size)

                             ;; 4. Verify results
                             (with-release [c0 (transfer! (view-batch tensor-c 0) (fge 2 3))
                                            c1 (transfer! (view-batch tensor-c 1) (fge 2 3))]

                                           (println "Verification Batch 0 (Should be true):" (= (seq a0-data) (seq (view-vctr c0))))
                                           (println "Verification Batch 1 (Should be true):" (= (seq a1-data) (seq (view-vctr c1))))

                                           (println "Result 0:\n" c0)
                                           (println "Result 1:\n" c1))))))

(ns batch-gemm
  (:require [uncomplicate.neanderthal.core :refer [view-vctr transfer! view-ge subvector]]
            [uncomplicate.neanderthal.internal.api :as api]
            [uncomplicate.neanderthal.native :refer [fv fge]]
            [uncomplicate.neanderthal.internal.cpp.mkl.factory :refer [mkl-float]]
            [uncomplicate.commons.core :refer [Info info with-release]]
            [neanderthal-blas-like.batch :refer [mm-batch-strided! batch-ge batch-ge-slab]]
            [neanderthal-blas-like.be.mkl]))

(let [fact mkl-float
      batch-size 2]
  (with-release [slab-a (batch-ge-slab fact 2 3 batch-size)
                 slab-b (batch-ge-slab fact 3 3 batch-size)
                 slab-c (batch-ge-slab fact 2 3 batch-size)]

    ;; 1. Setup A: Batch of 2 (2x3 matrices) with distinct values
    ;; Matrix 0: [1 3 5]
    ;;           [2 4 6] (Column-major)
    ;; Matrix 1: [7 9 11]
    ;;           [8 10 12] (Column-major)
    (let [a0-data (fv [1.0 2.0 3.0 4.0 5.0 6.0])
          a1-data (fv [7.0 8.0 9.0 10.0 11.0 12.0])]
      (transfer! a0-data (batch-ge slab-a 2 3 0))
      (transfer! a1-data (batch-ge slab-a 2 3 1))

      ;; 2. Setup B: Batch of 2 (3x3 Identity matrices)
      (let [identity-3x3 (fv [1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0])]
        (transfer! identity-3x3 (batch-ge slab-b 3 3 0))
        (transfer! identity-3x3 (batch-ge slab-b 3 3 1))

        ;; 3. Batched Multiply: C = A * B
        (mm-batch-strided! 1.0
                           (view-ge (view-vctr slab-a) 2 3) 6 ;; stride-a: 2x3
                           (view-ge (view-vctr slab-b) 3 3) 9 ;; stride-b: 3x3
                           0.0
                           (view-ge (view-vctr slab-c) 2 3) 6 ;; stride-c: 2x3
                           batch-size)

        ;; 4. Verify results: result0 should equal a0-data, result1 should equal a1-data
        (with-release [c0 (transfer! (batch-ge slab-c 2 3 0) (fge 2 3))
                       c1 (transfer! (batch-ge slab-c 2 3 1) (fge 2 3))]

          (println "Verification Batch 0 (Should be true):" (= (seq a0-data) (seq (view-vctr c0))))
          (println "Verification Batch 1 (Should be true):" (= (seq a1-data) (seq (view-vctr c1))))

          (println "Result 0:\n" c0)
          (println "Result 1:\n" c1))))))

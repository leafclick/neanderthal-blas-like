;;   Copyright (c) Kamil Toman. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 2.0 (https://opensource.org/licenses/EPL-2.0) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns neanderthal-blas-like.batch-test-common-ge
  (:require [midje.sweet :refer [facts fact roughly just =>]]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal.core :refer [ge mm! axpy! entry! transfer! trans]]
            [uncomplicate.neanderthal.internal.api :refer [engine view-vctr]]
            [neanderthal-blas-like.internal.api :refer [gemm-batch-strided]]
            [neanderthal-blas-like.batch :refer [batch-ge-slab batch-ge]]))

(defn test-gemm-batch-strided-basic [native-factory factory]
  (facts
    "gemm-batch-strided: basic correctness against individual mm!"
    (let [m 3 k 4 n 2
          batch-size 5
          stride-a (* m k)
          stride-b (* k n)
          stride-c (* m n)]
      ;; prepare data on native (CPU) side
      (with-release [na-slab (batch-ge-slab native-factory m k batch-size)
                     nb-slab (batch-ge-slab native-factory k n batch-size)
                     nc-ref  (batch-ge-slab native-factory m n batch-size)]
        (dotimes [b batch-size]
          (let [a-i (batch-ge na-slab m k b)
                b-i (batch-ge nb-slab k n b)]
            (dotimes [r m]
              (dotimes [c k]
                (entry! a-i r c (+ 1.0 (* 0.1 b) (* 0.01 r) (* 0.001 c)))))
            (dotimes [r k]
              (dotimes [c n]
                (entry! b-i r c (- 2.0 (* 0.05 b) (* 0.02 r) (* 0.003 c)))))))
        ;; compute reference on native
        (dotimes [b batch-size]
          (mm! 1.0 (batch-ge na-slab m k b) (batch-ge nb-slab k n b)
               0.0 (batch-ge nc-ref m n b)))
        ;; transfer to target factory and run batched op
        (with-release [a-slab (transfer! na-slab (batch-ge-slab factory m k batch-size))
                       b-slab (transfer! nb-slab (batch-ge-slab factory k n batch-size))
                       c-slab (batch-ge-slab factory m n batch-size)]
          (let [a0 (batch-ge a-slab m k 0)
                b0 (batch-ge b-slab k n 0)
                c0 (batch-ge c-slab m n 0)]
            (gemm-batch-strided (engine a0) 1.0
              a0 stride-a b0 stride-b
              0.0 c0 stride-c batch-size))
          ;; transfer results back and compare
          (dotimes [b batch-size]
            (with-release [ci (transfer! (batch-ge c-slab m n b)
                                         (ge native-factory m n))
                           ri (batch-ge nc-ref m n b)]
              (fact (seq (view-vctr ci)) => (just (map #(roughly %1 1e-5) (seq (view-vctr ri))))))))))))

(defn test-gemm-batch-strided-alpha-beta [native-factory factory]
  (facts
    "gemm-batch-strided: non-trivial alpha and beta"
    (let [m 2 k 3 n 2
          batch-size 3
          stride-a (* m k)
          stride-b (* k n)
          stride-c (* m n)
          alpha 2.5
          beta 0.75]
      ;; prepare data on native side
      (with-release [na-slab (batch-ge-slab native-factory m k batch-size)
                     nb-slab (batch-ge-slab native-factory k n batch-size)
                     nc-init (batch-ge-slab native-factory m n batch-size)
                     nc-ref  (batch-ge-slab native-factory m n batch-size)]
        (dotimes [b batch-size]
          (let [a-i (batch-ge na-slab m k b)
                b-i (batch-ge nb-slab k n b)
                c-i (batch-ge nc-init m n b)
                r-i (batch-ge nc-ref m n b)]
            (dotimes [r m]
              (dotimes [c k]
                (entry! a-i r c (+ 1.0 (* 0.5 b) (* 0.1 r) (* 0.2 c)))))
            (dotimes [r k]
              (dotimes [c n]
                (entry! b-i r c (- 3.0 (* 0.3 b) (* 0.1 r)))))
            (dotimes [r m]
              (dotimes [c n]
                (entry! c-i r c (+ 10.0 (* 0.7 b) (* 0.3 r) (* 0.2 c)))
                (entry! r-i r c (+ 10.0 (* 0.7 b) (* 0.3 r) (* 0.2 c)))))))
        ;; compute reference on native
        (dotimes [b batch-size]
          (mm! alpha (batch-ge na-slab m k b) (batch-ge nb-slab k n b)
               beta (batch-ge nc-ref m n b)))
        ;; transfer to target factory and run batched op
        (with-release [a-slab (transfer! na-slab (batch-ge-slab factory m k batch-size))
                       b-slab (transfer! nb-slab (batch-ge-slab factory k n batch-size))
                       c-slab (transfer! nc-init (batch-ge-slab factory m n batch-size))]
          (let [a0 (batch-ge a-slab m k 0)
                b0 (batch-ge b-slab k n 0)
                c0 (batch-ge c-slab m n 0)]
            (gemm-batch-strided (engine a0) alpha
              a0 stride-a b0 stride-b
              beta c0 stride-c batch-size))
          ;; transfer results back and compare
          (dotimes [b batch-size]
            (with-release [ci (transfer! (batch-ge c-slab m n b)
                                         (ge native-factory m n))
                           ri (batch-ge nc-ref m n b)]
              (fact (seq (view-vctr ci)) => (just (map #(roughly %1 1e-5) (seq (view-vctr ri))))))))))))

(defn test-gemm-batch-strided-single [native-factory factory]
  (facts
    "gemm-batch-strided: simple hand-verified two-batch test"
    ;; Batch 0: A0 = [[1 2] [3 4]], B0 = [[5] [6]]
    ;;   C0 = A0*B0 = [[1*5+2*6] [3*5+4*6]] = [[17] [39]]
    ;; Batch 1: A1 = [[1 0] [0 1]], B1 = [[7] [8]]
    ;;   C1 = A1*B1 = [[7] [8]]
    ;; With alpha=1.0, beta=0.0
    (let [m 2 k 2 n 1
          batch-size 2
          stride-a (* m k)
          stride-b (* k n)
          stride-c (* m n)]
      (with-release [na-slab (batch-ge-slab native-factory m k batch-size)
                     nb-slab (batch-ge-slab native-factory k n batch-size)
                     nc-init (batch-ge-slab native-factory m n batch-size)]
        ;; A0 = [[1 2] [3 4]]  (col-major: col0=[1,3] col1=[2,4])
        (let [a0 (batch-ge na-slab m k 0)]
          (entry! a0 0 0 1.0) (entry! a0 1 0 3.0)
          (entry! a0 0 1 2.0) (entry! a0 1 1 4.0))
        ;; A1 = identity
        (let [a1 (batch-ge na-slab m k 1)]
          (entry! a1 0 0 1.0) (entry! a1 1 0 0.0)
          (entry! a1 0 1 0.0) (entry! a1 1 1 1.0))
        ;; B0 = [[5] [6]]
        (let [b0 (batch-ge nb-slab k n 0)]
          (entry! b0 0 0 5.0) (entry! b0 1 0 6.0))
        ;; B1 = [[7] [8]]
        (let [b1 (batch-ge nb-slab k n 1)]
          (entry! b1 0 0 7.0) (entry! b1 1 0 8.0))
        ;; C init to 999 so a no-op is unmistakable
        (dotimes [b batch-size]
          (let [ci (batch-ge nc-init m n b)]
            (dotimes [r m] (dotimes [c n] (entry! ci r c 999.0)))))

        (with-release [a-slab (transfer! na-slab (batch-ge-slab factory m k batch-size))
                       b-slab (transfer! nb-slab (batch-ge-slab factory k n batch-size))
                       c-slab (transfer! nc-init (batch-ge-slab factory m n batch-size))]
          (let [a0 (batch-ge a-slab m k 0)
                b0 (batch-ge b-slab k n 0)
                c0 (batch-ge c-slab m n 0)]
            (gemm-batch-strided (engine a0) 1.0
              a0 stride-a b0 stride-b
              0.0 c0 stride-c batch-size))
          (with-release [c-res (transfer! c-slab (batch-ge-slab native-factory m n batch-size))]
            ;; C0 = [[17] [39]]
            (fact "batch 0" (seq (view-vctr (batch-ge c-res m n 0)))
                  => (just [(roughly 17.0 1e-5) (roughly 39.0 1e-5)]))
            ;; C1 = [[7] [8]]
            (fact "batch 1" (seq (view-vctr (batch-ge c-res m n 1)))
                  => (just [(roughly 7.0 1e-5) (roughly 8.0 1e-5)]))))))))



(defn test-gemm-batch-strided-square [native-factory factory]
  (facts
    "gemm-batch-strided: square matrices"
    (let [n 3
          batch-size 4
          stride (* n n)]
      ;; prepare data on native side
      (with-release [na-slab (batch-ge-slab native-factory n n batch-size)
                     nb-slab (batch-ge-slab native-factory n n batch-size)
                     nc-ref  (batch-ge-slab native-factory n n batch-size)]
        (dotimes [b batch-size]
          (let [a-i (batch-ge na-slab n n b)
                b-i (batch-ge nb-slab n n b)]
            (dotimes [r n]
              (dotimes [c n]
                (entry! a-i r c (if (= r c) (+ 1.0 (* 0.1 b)) 0.0))
                (entry! b-i r c (+ r (* 10 c) (* 0.5 b)))))))
        ;; compute reference on native
        (dotimes [b batch-size]
          (mm! 1.0 (batch-ge na-slab n n b) (batch-ge nb-slab n n b)
               0.0 (batch-ge nc-ref n n b)))
        ;; transfer to target factory and run batched op
        (with-release [a-slab (transfer! na-slab (batch-ge-slab factory n n batch-size))
                       b-slab (transfer! nb-slab (batch-ge-slab factory n n batch-size))
                       c-slab (batch-ge-slab factory n n batch-size)]
          (let [a0 (batch-ge a-slab n n 0)
                b0 (batch-ge b-slab n n 0)
                c0 (batch-ge c-slab n n 0)]
            (gemm-batch-strided (engine a0) 1.0
              a0 stride b0 stride
              0.0 c0 stride batch-size))
          ;; transfer results back and compare
          (dotimes [b batch-size]
            (with-release [ci (transfer! (batch-ge c-slab n n b)
                                         (ge native-factory n n))
                           ri (batch-ge nc-ref n n b)]
              (fact (seq (view-vctr ci)) => (just (map #(roughly %1 1e-5) (seq (view-vctr ri))))))))))))

(defn test-gemm-batch-strided-transposed [native-factory factory]
  (facts
    "gemm-batch-strided: with transposed matrices"
    (let [m 3 k 4 n 2
          batch-size 2
          stride-a (* m k)
          stride-b (* k n)
          stride-c (* m n)]
      ;; prepare data on native (CPU) side
      (with-release [na-slab (batch-ge-slab native-factory k m batch-size)
                     nb-slab (batch-ge-slab native-factory n k batch-size)
                     nc-ref  (batch-ge-slab native-factory m n batch-size)]
        (dotimes [b batch-size]
          (let [a-i (batch-ge na-slab k m b)
                b-i (batch-ge nb-slab n k b)]
            (dotimes [r k]
              (dotimes [c m]
                (entry! a-i r c (+ 1.0 (* 0.1 b) (* 0.01 r) (* 0.001 c)))))
            (dotimes [r n]
              (dotimes [c k]
                (entry! b-i r c (- 2.0 (* 0.05 b) (* 0.02 r) (* 0.003 c)))))))
        ;; compute reference on native
        (dotimes [b batch-size]
          (mm! 1.0 (trans (batch-ge na-slab k m b)) (trans (batch-ge nb-slab n k b))
               0.0 (batch-ge nc-ref m n b)))
        ;; transfer to target factory and run batched op
        (with-release [a-slab (transfer! na-slab (batch-ge-slab factory k m batch-size))
                       b-slab (transfer! nb-slab (batch-ge-slab factory n k batch-size))
                       c-slab (batch-ge-slab factory m n batch-size)]
          (let [a0 (trans (batch-ge a-slab k m 0))
                b0 (trans (batch-ge b-slab n k 0))
                c0 (batch-ge c-slab m n 0)]
            (gemm-batch-strided (engine a0) 1.0
              a0 stride-a b0 stride-b
              0.0 c0 stride-c batch-size))
          ;; transfer results back and compare
          (dotimes [b batch-size]
            (with-release [ci (transfer! (batch-ge c-slab m n b)
                                         (ge native-factory m n))
                           ri (batch-ge nc-ref m n b)]
              (fact (seq (view-vctr ci)) => (just (map #(roughly %1 1e-5) (seq (view-vctr ri))))))))))))


(defn run-all-tests [native-factory factory]
  (test-gemm-batch-strided-basic native-factory factory)
  (test-gemm-batch-strided-alpha-beta native-factory factory)
  (test-gemm-batch-strided-single native-factory factory)
  (test-gemm-batch-strided-square native-factory factory)
  (test-gemm-batch-strided-transposed native-factory factory))

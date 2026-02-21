;;   Copyright (c) Kamil Toman. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 2.0 (https://opensource.org/licenses/EPL-2.0) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

;; ... existing code ...
(ns neanderthal-blas-like.batch-test-common-vctr
  (:require [midje.sweet :refer [facts fact roughly just =>]]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal.core :refer [vctr subvector mm! axpy! entry! transfer!]]
            [uncomplicate.neanderthal.internal.api :refer [engine view-vctr]]
            [neanderthal-blas-like.internal.api :refer [axpy-batch-strided]]
            [neanderthal-blas-like.batch :refer [batch-vctr-slab batch-vctr]]))

(defn test-axpy-batch-strided-basic [native-factory factory]
  (facts
    "axpy-batch-strided: basic correctness against individual axpy!"
    (let [n 10
          batch-size 5
          stride-x n
          stride-y n]
      ;; prepare data on native (CPU) side
      (with-release [nx-slab (batch-vctr-slab native-factory n batch-size)
                     ny-ref (batch-vctr-slab native-factory n batch-size)
                     ny-init (batch-vctr-slab native-factory n batch-size)]
        (dotimes [b batch-size]
          (let [x-i (batch-vctr nx-slab n b)
                y-i (batch-vctr ny-init n b)
                r-i (batch-vctr ny-ref n b)]
            (dotimes [i n]
              (entry! x-i i (+ 1.0 (* 0.1 b) (* 0.01 i)))
              (entry! y-i i (- 2.0 (* 0.05 b) (* 0.02 i)))
              (entry! r-i i (- 2.0 (* 0.05 b) (* 0.02 i))))))
        ;; compute reference on native
        (dotimes [b batch-size]
          (axpy! 1.0 (batch-vctr nx-slab n b) (batch-vctr ny-ref n b)))
        ;; transfer to target factory and run batched op
        (with-release [x-slab (transfer! nx-slab (batch-vctr-slab factory n batch-size))
                       y-slab (transfer! ny-init (batch-vctr-slab factory n batch-size))]
          (axpy-batch-strided (engine (batch-vctr x-slab n 0)) n 1.0
                              x-slab stride-x y-slab stride-y batch-size)
          ;; transfer results back and compare
          (dotimes [b batch-size]
            (with-release [yi (transfer! (batch-vctr y-slab n b)
                                         (vctr native-factory n))
                           ri (batch-vctr ny-ref n b)]
              (fact (seq (view-vctr yi)) => (just (map #(roughly %1 1e-5) (seq (view-vctr ri))))))))))))

(defn test-axpy-batch-strided-alpha [native-factory factory]
  (facts
    "axpy-batch-strided: non-trivial alpha"
    (let [n 8
          batch-size 4
          stride-x n
          stride-y n
          alpha 2.5]
      ;; prepare data on native (CPU) side
      (with-release [nx-slab (batch-vctr-slab native-factory n batch-size)
                     ny-ref (batch-vctr-slab native-factory n batch-size)
                     ny-init (batch-vctr-slab native-factory n batch-size)]
        (dotimes [b batch-size]
          (let [x-i (batch-vctr nx-slab n b)
                y-i (batch-vctr ny-init n b)
                r-i (batch-vctr ny-ref n b)]
            (dotimes [i n]
              (entry! x-i i (+ 1.5 (* 0.2 b) (* 0.05 i)))
              (entry! y-i i (- 3.0 (* 0.1 b) (* 0.03 i)))
              (entry! r-i i (- 3.0 (* 0.1 b) (* 0.03 i))))))
        ;; compute reference on native
        (dotimes [b batch-size]
          (axpy! alpha (batch-vctr nx-slab n b) (batch-vctr ny-ref n b)))
        ;; transfer to target factory and run batched op
        (with-release [x-slab (transfer! nx-slab (batch-vctr-slab factory n batch-size))
                       y-slab (transfer! ny-init (batch-vctr-slab factory n batch-size))]
          (axpy-batch-strided (engine (batch-vctr x-slab n 0)) n alpha
                              x-slab stride-x y-slab stride-y batch-size)
          ;; transfer results back and compare
          (dotimes [b batch-size]
            (with-release [yi (transfer! (batch-vctr y-slab n b)
                                         (vctr native-factory n))
                           ri (batch-vctr ny-ref n b)]
              (fact (seq (view-vctr yi)) => (just (map #(roughly %1 1e-5) (seq (view-vctr ri))))))))))))

(defn test-axpy-batch-strided-single [native-factory factory]
  (facts
    "axpy-batch-strided: simple hand-verified two-batch test"
    (with-release [x-slab (batch-vctr-slab native-factory 4 2)
                   y-slab (batch-vctr-slab native-factory 4 2)]
      ;; x = [1 2 3 4 | 5 6 7 8],  y = [10 20 30 40 | 50 60 70 80]
      (doseq [[i v] (map-indexed vector [1 2 3 4 5 6 7 8])]
        (entry! x-slab i v))
      (doseq [[i v] (map-indexed vector [10 20 30 40 50 60 70 80])]
        (entry! y-slab i v))
      (with-release [x-dev (transfer! x-slab (batch-vctr-slab factory 4 2))
                     y-dev (transfer! y-slab (batch-vctr-slab factory 4 2))]
        (axpy-batch-strided (engine (batch-vctr x-dev 4 0)) 4 2.0
                            x-dev 4 y-dev 4 2)
        (with-release [y-res (transfer! y-dev (batch-vctr-slab native-factory 4 2))]
          ;; y0 = 2*[1 2 3 4] + [10 20 30 40] = [12 24 36 48]
          (fact "batch 0" (seq (view-vctr (batch-vctr y-res 4 0)))
                => (just [(roughly 12.0 1e-5) (roughly 24.0 1e-5)
                          (roughly 36.0 1e-5) (roughly 48.0 1e-5)]))
          ;; y1 = 2*[5 6 7 8] + [50 60 70 80] = [60 72 84 96]
          (fact "batch 1" (seq (view-vctr (batch-vctr y-res 4 1)))
                => (just [(roughly 60.0 1e-5) (roughly 72.0 1e-5)
                          (roughly 84.0 1e-5) (roughly 96.0 1e-5)])))))))

(defn test-axpy-batch-strided-non-contiguous [native-factory factory]
  (facts
    "axpy-batch-strided: with non-contiguous vector strides (gap between batches)"
    (let [n 5
          batch-size 3
          stride-x (* n 2)
          stride-y (* n 3)
          alpha 1.5]
      ;; Create a slab that is larger than n * batch-size
      (with-release [nx-slab (vctr native-factory (* stride-x batch-size))
                     ny-ref (vctr native-factory (* stride-y batch-size))
                     ny-init (vctr native-factory (* stride-y batch-size))]
        (dotimes [b batch-size]
          (let [x-i (batch-vctr nx-slab stride-x b)         ; extract with custom offset step
                y-i (batch-vctr ny-init stride-y b)
                r-i (batch-vctr ny-ref stride-y b)]
            (dotimes [i n]
              (entry! x-i i (+ 0.1 (* 0.1 b) (* 0.01 i)))
              (entry! y-i i (- 0.5 (* 0.05 b) (* 0.02 i)))
              (entry! r-i i (- 0.5 (* 0.05 b) (* 0.02 i))))))
        ;; compute reference on native
        (dotimes [b batch-size]
          (axpy! alpha
                 (subvector nx-slab (* b stride-x) n)
                 (subvector ny-ref (* b stride-y) n)))
        ;; transfer to target factory and run batched op
        (with-release [x-slab (transfer! nx-slab (vctr factory (* stride-x batch-size)))
                       y-slab (transfer! ny-init (vctr factory (* stride-y batch-size)))]
          (axpy-batch-strided (engine (subvector x-slab 0 n)) n alpha
                              x-slab stride-x y-slab stride-y batch-size)
          ;; transfer results back and compare
          (dotimes [b batch-size]
            (with-release [yi (transfer! (subvector y-slab (* b stride-y) n)
                                         (vctr native-factory n))
                           ri (subvector ny-ref (* b stride-y) n)]
              (fact (seq (view-vctr yi)) => (just (map #(roughly %1 1e-5) (seq (view-vctr ri))))))))))))

(defn run-all-tests [native-factory factory]
  (test-axpy-batch-strided-basic native-factory factory)
  (test-axpy-batch-strided-alpha native-factory factory)
  (test-axpy-batch-strided-single native-factory factory)
  (test-axpy-batch-strided-non-contiguous native-factory factory))

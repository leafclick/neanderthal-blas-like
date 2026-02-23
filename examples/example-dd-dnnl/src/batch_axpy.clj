(ns batch_axpy)
(ns batch-axpy
    (:require [uncomplicate.neanderthal.core :refer [view-vctr transfer! axpy!]]
      [uncomplicate.neanderthal.native :refer [fv]]
      [uncomplicate.diamond.tensor :as tz :refer [view-tz offset!]]
      [uncomplicate.diamond.internal.dnnl.factory :refer [dnnl-factory]]
      [uncomplicate.commons.core :refer [with-release]]
      [neanderthal-blas-like.batch :refer [axpy-batch-strided!]]
      [neanderthal-blas-like.be.mkl]))

(defn batch-stride [tz] (first (tz/layout tz)))

(defn view-batch [tz idx]
      (let [sub (into [1] (rest (tz/shape tz)))]
           (-> (doto (tz/view-tz tz sub)
                     (tz/offset! (* idx (batch-stride tz))))
               (view-vctr))))

(let [fact (dnnl-factory)
      batch-size 2
      vec-len 4]
     (with-release [tensor-x (tz/tensor fact [batch-size vec-len] :float :nc)
                    tensor-y (tz/tensor fact [batch-size vec-len] :float :nc)]

                   ;; 1. Setup X: Batch of 2 vectors
                   ;; Vector 0: [1 2 3 4]
                   ;; Vector 1: [5 6 7 8]
                   (let [x0-data (fv [1.0 2.0 3.0 4.0])
                         x1-data (fv [5.0 6.0 7.0 8.0])]
                        (transfer! x0-data (view-batch tensor-x 0))
                        (transfer! x1-data (view-batch tensor-x 1))

                        ;; 2. Setup Y: Batch of 2 vectors
                        ;; Vector 0: [10 20 30 40]
                        ;; Vector 1: [50 60 70 80]
                        (let [y0-data (fv [10.0 20.0 30.0 40.0])
                              y1-data (fv [50.0 60.0 70.0 80.0])]
                             (transfer! y0-data (view-batch tensor-y 0))
                             (transfer! y1-data (view-batch tensor-y 1))

                             ;; 3. Batched Vector Addition: Y_i = 2.0 * X_i + Y_i
                             (axpy-batch-strided! vec-len
                                                  2.0
                                                  (view-batch tensor-x 0) (batch-stride tensor-x)
                                                  (view-batch tensor-y 0) (batch-stride tensor-y)
                                                  batch-size)

                             ;; 4. Verify results: Y0 = 2.0*X0 + Y0, Y1 = 2.0*X1 + Y1
                             (with-release [expected-y0 (axpy! 2.0 x0-data (fv [10.0 20.0 30.0 40.0]))
                                            expected-y1 (axpy! 2.0 x1-data (fv [50.0 60.0 70.0 80.0]))
                                            y0-res (transfer! (view-batch tensor-y 0) (fv vec-len))
                                            y1-res (transfer! (view-batch tensor-y 1) (fv vec-len))]

                                           (println "Verification Batch 0 (Should be true):" (= (seq expected-y0) (seq y0-res)))
                                           (println "Verification Batch 1 (Should be true):" (= (seq expected-y1) (seq y1-res)))

                                           (println "Result Y0:\n" y0-res)
                                           (println "Result Y1:\n" y1-res))))))
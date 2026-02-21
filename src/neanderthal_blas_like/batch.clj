;;   Copyright (c) Kamil Toman. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 2.0 (https://opensource.org/licenses/EPL-2.0) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns neanderthal-blas-like.batch
  (:require [uncomplicate.neanderthal.core :refer [ge view-ge view-vctr subvector vctr]]
            [uncomplicate.neanderthal.internal.api :as api]
            [neanderthal-blas-like.internal.api :refer [BatchedBlas gemm-batch-strided axpy-batch-strided]]))

(defn batch-ge-slab
  "Create a contiguous buffer holding `batch-size` matrices of shape m×n,
  laid out consecutively.
  Returns the full GE slab viewed as (m, n*batch-size)."
  [factory ^long m ^long n ^long batch-size]
  (ge factory m (* n batch-size)))

(defn batch-ge
  "Extract the i-th matrix from a slab (column-major, stride = m*n)"
  [slab ^long m ^long n ^long i]
  (let [offset (* i m n)]
    (view-ge (subvector (view-vctr slab) offset (* m n)) m n)))

(defn batch-vctr-slab
  "Create a contiguous buffer holding `batch-size` vectors of size n,
  laid out consecutively.
  Returns the full VCTR slab viewed as a single vector."
  [factory ^long n ^long batch-size]
  (vctr factory (* n batch-size)))

(defn batch-vctr
  "Extract the i-th vector from a slab (stride = n)"
  [slab ^long n ^long i]
  (subvector slab (* i n) n))

(defn mm-batch-strided!
  "Batched matrix multiply: C_i = alpha * A_i * B_i + beta * C_i"
  [alpha a stride-a b stride-b beta c stride-c batch-size]
  (gemm-batch-strided (api/engine c) alpha a stride-a b stride-b beta c stride-c batch-size))

(defn axpy-batch-strided!
  "Batched vector addition: Y_i = alpha * X_i + Y_i
  n is the number of elements per vector in each batch."
  [n alpha x stride-x y stride-y batch-size]
  (axpy-batch-strided (api/engine y) n alpha x stride-x y stride-y batch-size))

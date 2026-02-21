;;   Copyright (c) Kamil Toman. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 2.0 (https://opensource.org/licenses/EPL-2.0) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns neanderthal-blas-like.internal.api)

;; ============ Batched API ====================================================

(defprotocol BatchedBlas
  (gemm-batch-strided [eng alpha a stride-a b stride-b beta c stride-c batch-size])
  (axpy-batch-strided [eng n alpha x stride-x y stride-y batch-size]))

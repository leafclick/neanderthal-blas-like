extern "C" {

#ifndef NUMBER
#define NUMBER float
#endif

#ifndef ACCUMULATOR
#define ACCUMULATOR float
#endif

    // Batched strided AXPY: y[i] = alpha * x[i] + y[i]
    // for batch_count batches, where batch b starts at offset b * stride_x (resp. stride_y)
    // Mimics MKL cblas_?axpy_batch_strided semantics.
    __global__ void axpy_batch_strided (const int n,
                                        const NUMBER alpha,
                                        const NUMBER* x, const int offset_x, const int incx, const long long stride_x,
                                        NUMBER* y, const int offset_y, const int incy, const long long stride_y,
                                        const int batch_count) {
        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = n * batch_count;
        if (gid < total) {
            const int batch = gid / n;
            const int idx   = gid % n;
            const long long ix = offset_x + batch * stride_x + idx * incx;
            const long long iy = offset_y + batch * stride_y + idx * incy;
            y[iy] = alpha * x[ix] + y[iy];
        }
    }
}
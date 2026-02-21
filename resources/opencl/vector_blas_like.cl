#ifndef REAL
#define REAL float
#endif

#ifndef WGS
#define WGS 256
#endif

__attribute__((work_group_size_hint(WGS, 1, 1)))
__kernel void axpy_batch_strided(const int n, const REAL alpha,
                               __global const REAL* x, const int offset_x, const int incx, const long stride_x,
                               __global REAL* y, const int offset_y, const int incy, const long stride_y,
                               const int batch_count) {
  const int gid = get_global_id(0);
  const int total = n * batch_count;
  if (gid < total) {
      const int batch = gid / n;
      const int idx   = gid % n;
      const long ix = offset_x + batch * stride_x + idx * incx;
      const long iy = offset_y + batch * stride_y + idx * incy;
      y[iy] = alpha * x[ix] + y[iy];
  }
}
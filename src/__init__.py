# Our matrices are small (at most a few hundred rows), where multithreaded
# OpenBLAS is far slower than one thread, especially on busy cluster login nodes.
from threadpoolctl import threadpool_limits

threadpool_limits(limits=1, user_api="blas")

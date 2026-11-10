#ifndef QL_THREADPOOL_H
#define QL_THREADPOOL_H

#include <stddef.h>

typedef struct threadpool threadpool;

threadpool *tp_create(int n_threads, char *err, size_t errlen);
void tp_free(threadpool *tp);
int tp_threads(const threadpool *tp);

/* Runs fn(arg, tid, nthreads) on every worker and waits for completion. */
int tp_run(threadpool *tp, void (*fn)(void *, int, int), void *arg);

#endif /* QL_THREADPOOL_H */

#define WIN32_LEAN_AND_MEAN
#include "threadpool.h"

#include <windows.h>

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    threadpool *tp;
    int tid;
    unsigned seen;
} worker_arg;

struct threadpool {
    int n;
    HANDLE *threads;
    worker_arg *wargs;
    CRITICAL_SECTION lock;
    CONDITION_VARIABLE cv_work;
    CONDITION_VARIABLE cv_done;
    unsigned gen;
    int stop;
    int done;
    void (*fn)(void *, int, int);
    void *arg;
};

static int terr(char *err, size_t errlen, const char *fmt, ...) {
    if (err && errlen) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errlen, fmt, ap);
        va_end(ap);
    }
    return -1;
}

/* One logical processor per physical core, when Windows exposes it. */
static DWORD_PTR g_core_mask[64];
static int g_core_count;

static void discover_cores(void) {
    if (g_core_count) return;
    DWORD len = 0;
    if (!GetLogicalProcessorInformation(NULL, &len)) {
        DWORD e = GetLastError();
        if (e != ERROR_INSUFFICIENT_BUFFER) return;
    }
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION *buf =
        (SYSTEM_LOGICAL_PROCESSOR_INFORMATION *)malloc(len);
    if (!buf) return;
    if (!GetLogicalProcessorInformation(buf, &len)) {
        free(buf);
        return;
    }
    DWORD count = len / sizeof(*buf);
    for (DWORD i = 0; i < count && g_core_count < 64; i++) {
        if (buf[i].Relationship == RelationProcessorCore)
            g_core_mask[g_core_count++] = buf[i].ProcessorMask;
    }
    free(buf);
}

static DWORD WINAPI worker_main(LPVOID p) {
    worker_arg *wa = (worker_arg *)p;
    threadpool *tp = wa->tp;
    discover_cores();
    if (g_core_count > 0)
        SetThreadAffinityMask(GetCurrentThread(),
                              g_core_mask[wa->tid % g_core_count]);

    for (;;) {
        EnterCriticalSection(&tp->lock);
        while (!tp->stop && tp->gen == wa->seen)
            SleepConditionVariableCS(&tp->cv_work, &tp->lock, INFINITE);
        if (tp->stop) {
            LeaveCriticalSection(&tp->lock);
            break;
        }
        wa->seen = tp->gen;
        void (*fn)(void *, int, int) = tp->fn;
        void *arg = tp->arg;
        LeaveCriticalSection(&tp->lock);

        fn(arg, wa->tid, tp->n);

        EnterCriticalSection(&tp->lock);
        tp->done++;
        if (tp->done == tp->n) WakeAllConditionVariable(&tp->cv_done);
        LeaveCriticalSection(&tp->lock);
    }
    return 0;
}

threadpool *tp_create(int n_threads, char *err, size_t errlen) {
    if (n_threads < 1 || n_threads > 64) {
        terr(err, errlen, "invalid thread count %d", n_threads);
        return NULL;
    }
    threadpool *tp = (threadpool *)calloc(1, sizeof(*tp));
    if (!tp) {
        terr(err, errlen, "out of memory");
        return NULL;
    }
    tp->n = n_threads;
    tp->threads = (HANDLE *)calloc((size_t)n_threads, sizeof(HANDLE));
    tp->wargs = (worker_arg *)calloc((size_t)n_threads, sizeof(worker_arg));
    if (!tp->threads || !tp->wargs) {
        terr(err, errlen, "out of memory");
        free(tp->threads);
        free(tp->wargs);
        free(tp);
        return NULL;
    }
    discover_cores();
    InitializeCriticalSection(&tp->lock);
    InitializeConditionVariable(&tp->cv_work);
    InitializeConditionVariable(&tp->cv_done);

    for (int i = 0; i < n_threads; i++) {
        tp->wargs[i].tp = tp;
        tp->wargs[i].tid = i;
        tp->wargs[i].seen = 0;
        tp->threads[i] = CreateThread(NULL, 0, worker_main, &tp->wargs[i], 0,
                                      NULL);
        if (!tp->threads[i]) {
            terr(err, errlen, "CreateThread failed (error %lu)",
                 (unsigned long)GetLastError());
            tp->stop = 1;
            tp->gen++;
            WakeAllConditionVariable(&tp->cv_work);
            for (int j = 0; j < i; j++) {
                WaitForSingleObject(tp->threads[j], INFINITE);
                CloseHandle(tp->threads[j]);
            }
            DeleteCriticalSection(&tp->lock);
            free(tp->threads);
            free(tp->wargs);
            free(tp);
            return NULL;
        }
    }
    return tp;
}

int tp_run(threadpool *tp, void (*fn)(void *, int, int), void *arg) {
    if (!tp) {
        fn(arg, 0, 1);
        return 0;
    }
    EnterCriticalSection(&tp->lock);
    tp->fn = fn;
    tp->arg = arg;
    tp->done = 0;
    tp->gen++;
    WakeAllConditionVariable(&tp->cv_work);
    while (tp->done < tp->n)
        SleepConditionVariableCS(&tp->cv_done, &tp->lock, INFINITE);
    LeaveCriticalSection(&tp->lock);
    return 0;
}

void tp_free(threadpool *tp) {
    if (!tp) return;
    if (tp->threads) {
        EnterCriticalSection(&tp->lock);
        tp->stop = 1;
        tp->gen++;
        WakeAllConditionVariable(&tp->cv_work);
        LeaveCriticalSection(&tp->lock);
        for (int i = 0; i < tp->n; i++) {
            if (tp->threads[i]) {
                WaitForSingleObject(tp->threads[i], INFINITE);
                CloseHandle(tp->threads[i]);
            }
        }
        DeleteCriticalSection(&tp->lock);
    }
    free(tp->threads);
    free(tp->wargs);
    free(tp);
}

int tp_threads(const threadpool *tp) { return tp ? tp->n : 1; }

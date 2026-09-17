#include <execinfo.h>
#include <signal.h>
#include <unistd.h>
static void debug_crash(int sig) {
    void *addresses[64];
    int count = backtrace(addresses, 64);
    backtrace_symbols_fd(addresses, count, 2);
    _exit(128 + sig);
}
static void __attribute__((constructor)) setup_debug_crash() {
    static char alternate_stack[65536];
    stack_t ss = {};
    ss.ss_sp = alternate_stack;
    ss.ss_size = sizeof(alternate_stack);
    sigaltstack(&ss, 0);
    struct sigaction sa = {};
    sa.sa_handler = debug_crash;
    sa.sa_flags = SA_ONSTACK;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, 0);
}

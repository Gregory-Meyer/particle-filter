#ifndef ALLOC_H
#define ALLOC_H

#include <stddef.h>

void *allocator(void *restrict arg, size_t size, size_t align);
void deallocator(void *restrict arg, void *restrict ptr, size_t size,
                 size_t align);

#endif

#include "alloc.h"

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

static bool is_power_of_two(size_t x);

static void *allocator(void *restrict arg, size_t size, size_t align) {
  (void)arg;
  assert(size > 0);
  assert(is_power_of_two(align));

  return aligned_alloc(align, size);
}

static bool is_power_of_two(size_t x) {
  return (x != 0) && ((x & (x - 1)) == 0);
}

static void deallocator(void *restrict arg, void *restrict ptr, size_t size,
                        size_t align) {
  (void)arg;
  assert(ptr);
  assert(size > 0);
  assert(is_power_of_two(align));

  free(ptr);
}

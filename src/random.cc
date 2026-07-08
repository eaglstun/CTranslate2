#include "ctranslate2/random.h"

#include <atomic>
#include <cstdint>

namespace ctranslate2 {

  constexpr unsigned int default_seed = static_cast<unsigned int>(-1);
  static std::atomic<unsigned int> g_seed(default_seed);
  static std::atomic<uint64_t> g_seed_epoch(0);

  void set_random_seed(const unsigned int seed) {
    g_seed = seed;
    // Bump the epoch so already-constructed thread-local generators reseed on their
    // next use: sampling draws happen on worker threads while set_random_seed is
    // typically called from the main thread.
    g_seed_epoch.fetch_add(1);
  }

  unsigned int get_random_seed() {
    return g_seed == default_seed ? std::random_device{}() : g_seed.load();
  }

  std::mt19937& get_random_generator() {
    static thread_local std::mt19937 generator(get_random_seed());
    static thread_local uint64_t seen_epoch = g_seed_epoch.load();
    const uint64_t epoch = g_seed_epoch.load();
    if (epoch != seen_epoch) {
      generator.seed(get_random_seed());
      seen_epoch = epoch;
    }
    return generator;
  }

}

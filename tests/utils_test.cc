#include <algorithm>
#include <cstdlib>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include "ctranslate2/utils.h"

#ifdef _OPENMP
#  include <omp.h>
#endif

using namespace ctranslate2;

namespace ctranslate2 {
  namespace cpu {
#ifndef _OPENMP
    size_t get_num_threads();
#endif
  }
}

TEST(UtilsTest, AutomaticNumThreads) {
  if (std::getenv("OMP_NUM_THREADS"))
    GTEST_SKIP() << "OMP_NUM_THREADS overrides the automatic default";

  set_num_threads(0);

#if defined(__APPLE__) && defined(CT2_ARM64_BUILD) && !defined(_OPENMP) && defined(CT2_WITH_RUY)
  constexpr size_t expected_num_threads = 1;
#else
  constexpr size_t default_num_threads = 4;
  const size_t max_num_threads = std::thread::hardware_concurrency();
  const size_t expected_num_threads = max_num_threads == 0
                                      ? default_num_threads
                                      : std::min(default_num_threads, max_num_threads);
#endif

#ifdef _OPENMP
  EXPECT_EQ(omp_get_max_threads(), expected_num_threads);
#else
  EXPECT_EQ(cpu::get_num_threads(), expected_num_threads);
#endif

  set_num_threads(1);
}

#if defined(__APPLE__) && !defined(_OPENMP)
TEST(UtilsTest, EnvironmentOverridesAutomaticNumThreads) {
  const char* current_value = std::getenv("OMP_NUM_THREADS");
  const bool had_value = current_value != nullptr;
  const std::string saved_value = current_value ? current_value : "";

  ASSERT_EQ(setenv("OMP_NUM_THREADS", "3", 1), 0);
  set_num_threads(0);
  EXPECT_EQ(cpu::get_num_threads(), 3);

  if (had_value)
    ASSERT_EQ(setenv("OMP_NUM_THREADS", saved_value.c_str(), 1), 0);
  else
    ASSERT_EQ(unsetenv("OMP_NUM_THREADS"), 0);
  set_num_threads(1);
}
#endif

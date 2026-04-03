#ifndef UTILS_H
#define UTILS_H
#include "common.h"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>

std::string state2str(const JointState &state);

template <class Vector>
inline std::string join(const Vector &array, bool high_precision = false) {
  std::ostringstream ss;
  for (size_t i = 0; i < array.size(); ++i) {
    if (i)
      ss << ", ";
    if (high_precision)
      ss << std::setprecision(16);
    ss << array[i];
  }
  return ss.str();
}

class RateLimiter {
public:
  explicit RateLimiter(double frequency_hz);

  // Wait until the next time step is reached
  void wait();

  // Check if enough time has passed for the next iteration (non-blocking)
  bool ready() const;

  // Reset the timer
  void reset();

  // Get the actual frequency achieved
  double get_actual_frequency() const;

  int get_iteration_count() const { return iteration_count_; }

private:
  std::chrono::microseconds period_;
  std::chrono::steady_clock::time_point last_time_;
  std::chrono::steady_clock::time_point start_time_;
  int iteration_count_;
};

#endif

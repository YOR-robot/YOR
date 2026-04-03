#include "utils.h"
#include <thread>

std::string state2str(const JointState &state) {
  std::string str = "";
  str += "pos:" + join(state.pos, false);
  str += " vel:" + join(state.vel, false);
  str += " torque:" + join(state.torque, false);
  str += " gripper_pos:" + std::to_string(state.gripper_pos);
  str += " timestamp:" + std::to_string(state.timestamp);
  str += "\n";
  return str;
}


// RateLimiter implementation
RateLimiter::RateLimiter(double frequency_hz)
    : period_(static_cast<int>(1e6 / frequency_hz)),
      last_time_(std::chrono::steady_clock::now()), start_time_(last_time_),
      iteration_count_(0) {}

void RateLimiter::wait() {
  auto now = std::chrono::steady_clock::now();
  auto elapsed = now - last_time_;

  if (elapsed < period_) {
    std::this_thread::sleep_for(period_ - elapsed);
    last_time_ += period_;
  } else {
    last_time_ = now;
  }

  iteration_count_++;
}

bool RateLimiter::ready() const {
  auto now = std::chrono::steady_clock::now();
  return (now - last_time_) >= period_;
}

void RateLimiter::reset() {
  last_time_ = std::chrono::steady_clock::now();
  start_time_ = last_time_;
  iteration_count_ = 0;
}

double RateLimiter::get_actual_frequency() const {
  if (iteration_count_ == 0)
    return 0.0;

  auto total_time = std::chrono::steady_clock::now() - start_time_;
  auto total_seconds = std::chrono::duration<double>(total_time).count();

  return iteration_count_ / total_seconds;
}

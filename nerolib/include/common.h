#ifndef COMMON_H
#define COMMON_H

#ifdef __APPLE__
#include <libkern/OSByteOrder.h>

#define htobe16(x) OSSwapHostToBigInt16(x)
#define htole16(x) OSSwapHostToLittleInt16(x)
#define be16toh(x) OSSwapBigToHostInt16(x)
#define le16toh(x) OSSwapLittleToHostInt16(x)

#define htobe32(x) OSSwapHostToBigInt32(x)
#define htole32(x) OSSwapHostToLittleInt32(x)
#define be32toh(x) OSSwapBigToHostInt32(x)
#define le32toh(x) OSSwapLittleToHostInt32(x)

#define htobe64(x) OSSwapHostToBigInt64(x)
#define htole64(x) OSSwapHostToLittleInt64(x)
#define be64toh(x) OSSwapBigToHostInt64(x)
#define le64toh(x) OSSwapLittleToHostInt64(x)

#else
#include <endian.h>
#endif

#include <array>
#include <thread>

#define MOTOR_DOF 7

// TODO: instead of JointState, create a struct for VecDoF
struct JointState {
  double timestamp = 0.0f;
  std::array<double, MOTOR_DOF> pos;
  std::array<double, MOTOR_DOF> vel;
  std::array<double, MOTOR_DOF> torque;
  double gripper_pos = 0.0f; // 0 for close, 1 for fully open
  double gripper_effort = 0.0f;

  JointState()
      : pos(std::array<double, MOTOR_DOF>{0.0}),
        vel(std::array<double, MOTOR_DOF>{0.0}),
        torque(std::array<double, MOTOR_DOF>{0.0}), gripper_pos(0.0f),
        gripper_effort(0.0f) {}
  JointState(std::array<double, MOTOR_DOF> pos,
             std::array<double, MOTOR_DOF> vel,
             std::array<double, MOTOR_DOF> torque, double gripper_pos)
      : pos(pos), vel(vel), torque(torque), gripper_pos(gripper_pos),
        gripper_effort(0.0f) {}

  JointState operator+(const JointState &other) const {
    JointState result;
    for (int i = 0; i < MOTOR_DOF; i++) {
      result.pos[i] = pos[i] + other.pos[i];
      result.vel[i] = vel[i] + other.vel[i];
      result.torque[i] = torque[i] + other.torque[i];
    }
    result.gripper_pos = gripper_pos + other.gripper_pos;
    return result;
  }
  JointState operator-(const JointState &other) const {
    JointState result;
    for (int i = 0; i < MOTOR_DOF; i++) {
      result.pos[i] = pos[i] - other.pos[i];
      result.vel[i] = vel[i] - other.vel[i];
      result.torque[i] = torque[i] - other.torque[i];
    }
    result.gripper_pos = gripper_pos - other.gripper_pos;
    return result;
  }
  JointState operator*(const double &scalar) const {
    JointState result;
    for (int i = 0; i < MOTOR_DOF; i++) {
      result.pos[i] = pos[i] * scalar;
      result.vel[i] = vel[i] * scalar;
      result.torque[i] = torque[i] * scalar;
    }
    result.gripper_pos = gripper_pos * scalar;
    return result;
  }
  JointState operator/(const double &scalar) const {
    JointState result;
    for (int i = 0; i < MOTOR_DOF; i++) {
      result.pos[i] = pos[i] / scalar;
      result.vel[i] = vel[i] / scalar;
      result.torque[i] = torque[i] / scalar;
    }
    result.gripper_pos = gripper_pos / scalar;
    return result;
  }
};

struct Gain {
  std::array<double, MOTOR_DOF> kp;
  std::array<double, MOTOR_DOF> kd;
  Gain()
      : kp(std::array<double, MOTOR_DOF>{0.0}),
        kd(std::array<double, MOTOR_DOF>{0.0}) {}
  Gain(std::array<double, MOTOR_DOF> kp, std::array<double, MOTOR_DOF> kd)
      : kp(kp), kd(kd) {
    if (kp.size() != kd.size())
      throw std::invalid_argument("Length of kp is not equal to kd.");
  }
  Gain operator+(const Gain &other) const {
    Gain result;
    for (int i = 0; i < MOTOR_DOF; i++) {
      result.kp[i] = kp[i] + other.kp[i];
      result.kd[i] = kd[i] + other.kd[i];
    }
    return result;
  }
  Gain operator*(const double &scalar) const {
    Gain result;
    for (int i = 0; i < MOTOR_DOF; i++) {
      result.kp[i] = kp[i] * scalar;
      result.kd[i] = kd[i] * scalar;
    }
    return result;
  }
};

inline void sleep_ms(int x) {
  std::this_thread::sleep_for(std::chrono::milliseconds(x));
}
inline void sleep_us(int x) {
  std::this_thread::sleep_for(std::chrono::microseconds(x));
}
inline std::chrono::milliseconds get_time_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch());
}
inline std::chrono::microseconds get_time_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now().time_since_epoch());
}

#endif // COMMON_H

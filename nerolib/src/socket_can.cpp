
#ifdef __linux__
#include "socket_can.h"
#include <cstring>
#include <linux/sockios.h>
#include <stdexcept>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

// Static function for receiver thread
static void *receiver_thread_function(void *arg) {
  SocketCAN *can = static_cast<SocketCAN *>(arg);
  can->receiver_thread_running = true;

  while (!can->terminate_receiver_thread) {
    can_frame_t frame;
    ssize_t nbytes = read(can->sockfd, &frame, sizeof(struct can_frame));

    if (nbytes < 0) {
      // Error reading
      if (errno != EAGAIN && errno != EWOULDBLOCK) {
        break;
      }
      continue;
    }

    if (nbytes < sizeof(struct can_frame)) {
      // Incomplete frame
      continue;
    }

    if (can->reception_handler) {
      can->reception_handler(&frame);
    }
  }

  can->receiver_thread_running = false;
  return nullptr;
}

SocketCAN::SocketCAN()
    : adapter_type(ADAPTER_SOCKETCAN), reception_handler(nullptr) {}

SocketCAN::~SocketCAN() {
  if (is_open()) {
    close();
  }
}

void SocketCAN::open(const char *interface_name) {
  // Create CAN socket
  sockfd = socket(PF_CAN, SOCK_RAW, CAN_RAW);
  if (sockfd < 0) {
    throw std::runtime_error("Failed to create CAN socket");
  }

  // Get interface index
  std::strncpy(if_request.ifr_name, interface_name, IFNAMSIZ - 1);
  if (ioctl(sockfd, SIOCGIFINDEX, &if_request) < 0) {
    ::close(sockfd);
    sockfd = -1;
    throw std::runtime_error("Failed to get interface index");
  }

  // Bind socket to the CAN interface
  addr.can_family = AF_CAN;
  addr.can_ifindex = if_request.ifr_ifindex;

  if (bind(sockfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
    ::close(sockfd);
    sockfd = -1;
    throw std::runtime_error("Failed to bind socket");
  }

  // Check if interface is UP
  if (ioctl(sockfd, SIOCGIFFLAGS, &if_request) < 0) {
    ::close(sockfd);
    sockfd = -1;
    throw std::runtime_error("Failed to get interface flags");
  }

  if (!(if_request.ifr_flags & IFF_UP)) {
    ::close(sockfd);
    sockfd = -1;
    throw std::runtime_error("Interface is not UP");
  }
}

void SocketCAN::close() {
  if (sockfd >= 0) {
    // Stop receiver thread if running
    if (receiver_thread_running) {
      terminate_receiver_thread = true;
      pthread_join(receiver_thread_id, nullptr);
    }

    ::close(sockfd);
    sockfd = -1;
  }
}

bool SocketCAN::is_open() { return sockfd >= 0; }

void SocketCAN::transmit(can_frame_t *frame) {
  if (!is_open()) {
    throw std::runtime_error("CAN socket not open");
  }

  ssize_t nbytes = write(sockfd, frame, sizeof(struct can_frame));
  if (nbytes != sizeof(struct can_frame)) {
    throw std::runtime_error("Failed to transmit CAN frame");
  }
}

void SocketCAN::start_receiver_thread() {
  if (!is_open()) {
    throw std::runtime_error("CAN socket not open");
  }

  if (receiver_thread_running) {
    return; // Thread already running
  }

  terminate_receiver_thread = false;

  int result = pthread_create(&receiver_thread_id, nullptr,
                              receiver_thread_function, this);
  if (result != 0) {
    throw std::runtime_error("Failed to create receiver thread");
  }
}
#else
// Dummy implementation for non-Linux systems
#include "socket_can.h"
#include <iostream>
#include <stdexcept>

SocketCAN::SocketCAN()
    : adapter_type(ADAPTER_SOCKETCAN), reception_handler(nullptr) {
  std::cerr << "WARNING: SocketCAN is not supported on this platform. Using "
               "dummy implementation."
            << std::endl;
}

SocketCAN::~SocketCAN() {}

void SocketCAN::open(const char *interface_name) {
  std::cerr << "WARNING: SocketCAN::open is a no-op on this platform."
            << std::endl;
  // Simulate open socket
  sockfd = 1;
}

void SocketCAN::close() { sockfd = -1; }

bool SocketCAN::is_open() { return sockfd >= 0; }

void SocketCAN::transmit(can_frame_t *frame) {
  // No-op
}

void SocketCAN::start_receiver_thread() {
  // No-op
}

#endif

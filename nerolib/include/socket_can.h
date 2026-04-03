#if (!defined(SOCKETCAN_H)) && (!defined(MINGW))
#define SOCKETCAN_H

#include <functional>
#include <net/if.h>
#include <pthread.h>
#include <stdbool.h>
#include <sys/socket.h>

#ifdef __linux__
#include <linux/can.h>
#include <linux/can/raw.h>
#else
// Mock definitions for non-Linux systems
typedef uint32_t canid_t;

struct can_frame {
  uint32_t can_id; /* 32 bit CAN_ID + EFF/RTR/ERR flags */
  uint8_t can_dlc; /* frame payload length in byte (0 .. 8) */
  uint8_t __pad;   /* padding */
  uint8_t __res0;  /* reserved / padding */
  uint8_t __res1;  /* reserved / padding */
  uint8_t data[8] __attribute__((aligned(8)));
};

#define CAN_EFF_FLAG 0x80000000U /* EFF/SFF is set in the MSB */
#define CAN_RTR_FLAG 0x40000000U /* remote transmission request */
#define CAN_ERR_FLAG 0x20000000U /* error message frame */

#define CAN_SFF_MASK 0x000007FFU /* standard frame format (SFF) */
#define CAN_EFF_MASK 0x1FFFFFFFU /* extended frame format (EFF) */
#define CAN_ERR_MASK 0x1FFFFFFFU /* omit EFF, RTR, ERR flags */

struct sockaddr_can {
  uint16_t can_family;
  int can_ifindex;
  union {
    /* transport protocol class address information (e.g. ISOTP) */
    struct {
      canid_t rx_id, tx_id;
    } tp;

    /* reserved for future usage */
  } can_addr;
};

#define SOL_CAN_BASE 100
#define SOL_CAN_RAW (SOL_CAN_BASE + 1)
#define CAN_RAW_FILTER 1
#define CAN_RAW_ERR_FILTER 2
#define CAN_RAW_LOOPBACK 3
#define CAN_RAW_RECV_OWN_MSGS 4
#define CAN_RAW_FD_FRAMES 5
#define CAN_RAW_JOIN_FILTERS 6

#endif

typedef struct can_frame can_frame_t;

typedef enum {
  ADAPTER_NONE,
  ADAPTER_SOCKETCAN,
  ADAPTER_SLCAN,
  ADAPTER_LOGFILE,
} can_adapter_t;

/**
 * Interface request structure used for socket ioctl's
 */
typedef struct ifreq interface_request_t;

/**
 * Socket address type for CAN sockets
 */
typedef struct sockaddr_can can_socket_address_t;

typedef void (*reception_handler_t)(can_frame_t *, void *);

/**
 * Facilitates frame transmission and reception via a CAN adapter
 */
class SocketCAN {

protected:
  can_adapter_t adapter_type;

private:
  interface_request_t if_request;

  can_socket_address_t addr;

  pthread_t receiver_thread_id;

public:
  std::function<void(can_frame_t *)> reception_handler;

  /**
   * CAN socket file descriptor
   */
  int sockfd = -1;

  /**
   * Request for the child thread to terminate
   */
  bool terminate_receiver_thread = false;

  bool receiver_thread_running = false;

  /** Constructor */
  SocketCAN();
  /** Destructor */
  ~SocketCAN();

  /**
   * Open and bind socket
   */
  void open(const char *);

  /**
   * Close and unbind socket
   */
  void close();

  /**
   * Returns whether the socket is open or closed
   *
   * @retval true     Socket is open
   * @retval false    Socket is closed
   */
  bool is_open();

  /**
   * Sends the referenced frame to the bus
   */
  void transmit(can_frame_t *);

  /**
   * Starts a new thread, that will wait for socket events
   */
  void start_receiver_thread();
};

#endif

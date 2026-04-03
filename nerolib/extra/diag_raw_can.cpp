// Quick diagnostic: print raw CAN bytes from high-speed feedback
// Run: conda run -n nerolib ./diag_raw_can can_left

#include "socket_can.h"
#include <cstdint>
#include <cstring>
#include <endian.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>

int main(int argc, char* argv[]) {
    std::string iface = (argc > 1) ? argv[1] : "can0";
    std::cout << "Listening on " << iface << " (printing raw high-speed 0x251-0x257 frames)...\n";

    SocketCAN can;
    can.reception_handler = [](can_frame_t* frame) {
        if (frame->can_id >= 0x251 && frame->can_id <= 0x257) {
            int motor = frame->can_id - 0x251;

            // Decode as per current nerolib (uint16 current bytes 2-3)
            int16_t  vel_raw;  std::memcpy(&vel_raw,  frame->data,     2); vel_raw  = be16toh(vel_raw);
            uint16_t cur_uint; std::memcpy(&cur_uint, frame->data + 2, 2); cur_uint = be16toh(cur_uint);
            int16_t  cur_int;  std::memcpy(&cur_int,  frame->data + 2, 2); cur_int  = be16toh(cur_int);
            int32_t  pos_raw;  std::memcpy(&pos_raw,  frame->data + 4, 4); pos_raw  = be32toh(pos_raw);

            float vel = vel_raw / 1000.0f;
            float pos = pos_raw / 1000.0f;
            float tau_uint = cur_uint / 1000.0f * 1.18125f;
            float tau_int  = cur_int  / 1000.0f * 1.18125f;

            std::cout << "J" << motor
                      << "  pos=" << std::setw(8) << pos
                      << "  vel=" << std::setw(8) << vel
                      << "  cur_raw=0x" << std::hex << std::setw(4) << std::setfill('0') << cur_uint
                      << std::dec << std::setfill(' ')
                      << "  cur_uint_A=" << std::setw(8) << (cur_uint / 1000.0f)
                      << "  tau(uint)=" << std::setw(8) << tau_uint
                      << "  cur_int_A=" << std::setw(8)  << (cur_int / 1000.0f)
                      << "  tau(int)="  << std::setw(8)  << tau_int
                      << "\n";
        }
    };
    can.open(iface.c_str());
    can.start_receiver_thread();

    // Print for 3 seconds then exit
    std::this_thread::sleep_for(std::chrono::seconds(3));
    std::cout << "Done.\n";
    return 0;
}

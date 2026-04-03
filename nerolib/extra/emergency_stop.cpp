#include "nero_interface.h"

int main() {
  NeroInterface nero_interface("can0");
  nero_interface.set_emergency_stop(EmergencyStop::STOP);
  return 0;
}
#include "solver.h"
#include <iostream>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <urdf_path>\n";
        return 1;
    }
    
    std::string urdf_path = argv[1];
    InverseDynamicsSolver solver(urdf_path);
    
    Eigen::VectorXd q(MOTOR_DOF);
    q << 0.5, -0.3, 0.2, 1.0, -0.5, 0.4, 0.1;
    
    pinocchio::forwardKinematics(solver.model_, solver.data_, q);
    pinocchio::updateFramePlacements(solver.model_, solver.data_);
    
    std::cout << "--- URDF Forward Kinematics ---\n";
    for (auto name : {"gripper_base", "link7", "link8", "dummy_link"}) {
        if (solver.model_.existFrame(name)) {
            auto frame_id = solver.model_.getFrameId(name);
            auto pos = solver.data_.oMf[frame_id].translation();
            std::cout << "Frame '" << name << "' pos: " 
                      << pos[0] << ", " << pos[1] << ", " << pos[2] << "\n";
        }
    }
    return 0;
}

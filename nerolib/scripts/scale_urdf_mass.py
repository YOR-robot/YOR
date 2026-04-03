#!/usr/bin/env python3
"""
Scale URDF Masses
===================
Scales the mass of all links in a URDF file by a given factor.
This is used to globally reduce gravity compensation feedforward torques
without modifying C++ code.
"""

import xml.etree.ElementTree as ET
import sys
import shutil

def scale_urdf_mass(input_path, output_path, scale_factor):
    tree = ET.parse(input_path)
    root = tree.getroot()

    for mass_tag in root.findall(".//mass"):
        old_val = float(mass_tag.get("value"))
        new_val = old_val * scale_factor
        mass_tag.set("value", f"{new_val:.5f}")

    # Write output preserving XML declaration
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"[ScaleURDF] Saved scaled URDF (factor={scale_factor}) to {output_path}")

if __name__ == "__main__":
    in_urdf  = "/home/cone-e/code/nerolib/urdf/nero_cone-e_left_fixed.urdf"
    out_urdf = "/home/cone-e/code/nerolib/urdf/nero_cone-e_left_scaled.urdf"

    # We scale mass by 70% to reduce gravity comp strength
    scale_factor = 0.70
    
    scale_urdf_mass(in_urdf, out_urdf, scale_factor)

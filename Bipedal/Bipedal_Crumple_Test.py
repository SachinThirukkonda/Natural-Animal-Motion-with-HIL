import time

import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path

xml_path = Path(__file__).parent / "agility_cassie" / "scene.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)
home_id = model.key("home").id
mujoco.mj_resetDataKeyframe(model, data, home_id)

mujoco.mj_kinematics(model, data)

# Bodies
for i in range(model.nbody):
    print(f"BODY  {i:2d}  {model.body(i).name:25s}  {data.xpos[i]}")

# Sites
for i in range(model.nsite):
    print(f"SITE  {i:2d}  {model.site(i).name:25s}  {data.site_xpos[i]}")

# Geoms
for i in range(model.ngeom):
    print(f"GEOM  {i:2d}  {model.geom(i).name:25s}  {data.geom_xpos[i]}")

for i in range(model.nsensor):
    print(f"SENSOR  {i:2d}  {model.sensor(i).name:25s}  {data.sensor(model.sensor(i).name).data}")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        remaining = model.opt.timestep - (time.time() - step_start)
        if remaining > 0:
            time.sleep(remaining)
#initial_pos = [ 4.49956e-03  0.00000e+00  4.97301e-01 -1.19970e+00  0.00000e+00
#  1.42671e+00 -2.25907e-06 -1.52439e+00  1.50645e+00 -1.59681e+00
# -4.49956e-03  0.00000e+00  4.97301e-01 -1.19970e+00  0.00000e+00
#  1.42671e+00  0.00000e+00 -1.52439e+00  1.50645e+00 -1.59681e+00]

import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path

xml_path = Path(__file__).parent / "agility_cassie" / "scene.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

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

    
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
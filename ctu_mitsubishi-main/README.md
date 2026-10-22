# Control of Mitsubishi RV6S robot at CIIRC CTU


## How to start the robot
- Turn the robot power using the white switch on the front panel of the control unit.
- Make sure robot is set to Automatic mode (or Ext. Automatic mode) - modes are controlled by key switch on the front panel.

## Quick example
```python
from ctu_mitsubishi import Rv6s, Rv6sGripper

gripper = Rv6sGripper()
gripper.open()
gripper.close()

robot = Rv6s()
robot.initialize()

q = robot.get_q()
q[0] += 0.1
robot.move_to_q(q)

robot.stop_robot()
robot.close_connection()
```

See more examples in the `examples` folder.
For the complete documentation see https://robotics-robot-stations.readthedocs.io/


## For developers
```
git clone git@github.com:CTURobotics/ctu_mitsubishi.git
cd ctu_mitsubishi
uv sync
uv pip install -e ".[dev]"
pre-commit install
```

cmake_minimum_required(VERSION 3.5)
project(pymoveit2)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)


# Install Python package
ament_python_install_package(pymoveit2)

# Install examples
set(EXAMPLES_DIR examples)
install(PROGRAMS
    ${EXAMPLES_DIR}/ex_collision_object.py
    ${EXAMPLES_DIR}/ex_joint_goal.py
    ${EXAMPLES_DIR}/ex_pose_goal.py
    ${EXAMPLES_DIR}/ex_servo.py
    ${EXAMPLES_DIR}/task3a.py
    ${EXAMPLES_DIR}/task2a1.py
    ${EXAMPLES_DIR}/task2a2.py
    ${EXAMPLES_DIR}/task1b.py
    ${EXAMPLES_DIR}/task2a.py
    ${EXAMPLES_DIR}/task2atfl.py
    ${EXAMPLES_DIR}/task1a.py
    ${EXAMPLES_DIR}/ex_collision_rack.py
    DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY examples/assets
  DESTINATION lib/${PROJECT_NAME}
)

# Setup the project
ament_package()



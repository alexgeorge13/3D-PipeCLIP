# src/prompts.py

CLASS_NAMES_GEOMETRIC_17 = [
    "a 3D point cloud of a Normal pipe. The walls form a continuous, smooth, uniform cylinder with no internal protrusions or breaks.",
    "a 3D point cloud of Settled deposits. A dense layer of points forming a flattened floor at the bottom of the cylindrical pipe.",
    "a 3D point cloud of a Displaced joint. There is a distinct, abrupt step or offset in the pipeline geometry where two cylindrical sections fail to align perfectly.",
    "a 3D point cloud of an Obstacle. A random cluster of points obstructing the interior volume of the pipeline, creating a sharp structural edge.",
    "a 3D point cloud of Attached deposits. A thick, irregular coating of points lining the internal surface wall of the cylinder.",
    "a 3D point cloud of Roots. An organic, web-like cluster of points penetrating through the wall and spreading inside the pipe volume.",
    "a 3D point cloud of Cracks, breaks, and collapses. The continuous cylindrical geometry is fractured, showing missing points, jagged edges, or inward buckling.",
    "a 3D point cloud of Deformation. The cross-section of the pipe is no longer perfectly circular, appearing compressed or oval due to external pressure.",
    "a 3D point cloud of Mechanical surface damage. External objects forcefully penetrating through the geometric boundary of the pipe wall.",
    "a 3D point cloud of Floating debris. Disconnected clusters of points suspended in the air or water level inside the pipe.",
    "a 3D point cloud of Chemical surface damage. The cylinder wall exhibits extreme roughness, pitting, and missing point density due to erosion.",
    "a 3D point cloud of Chiseled connection. A complete detachment or heavy geometric gap between the ends of two pipes.",
    "a 3D point cloud of Defective brickwork or masonry. A massive, blocky, square-like obstruction of points sitting heavily on the pipe floor.",
    "a 3D point cloud of Infiltration. Points showing external water or material bleeding inward through a specific fracture in the wall.",
    "a 3D point cloud of Undulation. A smooth, wave-like vertical shifting of the pipe floor forming continuous high and low elevation zones.",
    "a 3D point cloud of Intruding connection. A smaller cylindrical pipe structure intersecting laterally directly through the main pipe wall.",
    "a 3D point cloud of Interface material delamination. A thin, subtle strip or ribbon of material points peeling or hanging away from the internal joint of the cylinder walls."
]

# Mapping from AAU dataset labels (0, 1, 2, 3) to the indices in the 17-prompt list above
AAU_TO_IDX_MAP = {0: 0, 1: 2, 2: 12, 3: 16}
VALID_INDICES = [0, 2, 12, 16]
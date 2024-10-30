import pygmsh
import pyvista as pv
import numpy as np
import pygmsh
import pyvista as pv
import numpy as np

# Create the geometry and define a 3D domain
# with pygmsh.geo.Geometry() as geom:
#     # Define a box with specific corner points (x0, y0, z0) and (x1, y1, z1)
#     cube = geom.add_box(x0=0, x1=1, y0=0, y1=1, z0=0, z1=1, mesh_size=0.2)
#
#     # Generate the mesh
#     mesh = geom.generate_mesh()
#
#     vertices = mesh.points # Ensure vertices are integers
#     tetrahedrons = mesh.cells_dict["tetra"]
#
#
#
#
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
#
# # Plot each tetrahedron
# for tetra in tetrahedrons:
#     print(tetra)
#     # Get the vertices for this tetrahedron
#     tetra_points = vertices[tetra]
#
#     # Define the 4 triangular faces of the tetrahedron
#     faces = [
#         [tetra_points[0], tetra_points[1], tetra_points[2]],
#         [tetra_points[0], tetra_points[1], tetra_points[3]],
#         [tetra_points[0], tetra_points[2], tetra_points[3]],
#         [tetra_points[1], tetra_points[2], tetra_points[3]]
#     ]
#
#     # Create a Poly3DCollection for each face
#     poly = Poly3DCollection(faces, color="cyan", edgecolor="k", alpha=0.2)
#     ax.add_collection3d(poly)
#
# # Set plot limits
# ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
# ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
# ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
#
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
#
# plt.show()


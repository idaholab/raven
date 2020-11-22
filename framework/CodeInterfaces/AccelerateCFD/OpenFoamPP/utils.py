"""
utilities
"""


def calc_phase_surface_area(mesh, phi, face_area=None, omg=1.5):
    """
    calculate phase surface area for VOF
    :param mesh: FoamMesh object
    :param phi: vof data, numpy array
    :param face_area: face area, scalar or list or numpy array
    :param omg: power index
    :return: phase surface area
    """
    if face_area is not None:
        try:
            if len(face_area) == 1:
                face_area = [face_area[0]] * mesh.num_face
        except TypeError:
            face_area = [face_area] * mesh.num_face
    else:
        if mesh.face_areas is None:
            face_area = [1.] * mesh.num_face
        else:
            face_area = mesh.face_areas
    return sum([face_area[n]*abs(phi[mesh.owner[n]] - phi[mesh.neighbour[n]])**omg for n in range(mesh.num_inner_face)])

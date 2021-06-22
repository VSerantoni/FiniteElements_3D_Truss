# Problem 3D - Truss

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def formStiffness3Dtruss(GDof, numberElements, elementNodes, xx, yy, zz, E, A):
    stiffness = np.zeros([GDof, GDof])

    # Computation of the stiffness matrix
    for e in range(0, numberElements):
        indice = elementNodes[e, :]
        # calcul x2-x1 et y2-y1
        x2 = xx[indice[1]]-xx[indice[0]]
        y2 = yy[indice[1]] - yy[indice[0]]
        z2 = zz[indice[1]] - zz[indice[0]]
        # length of element and cos/sin angle
        Le = np.sqrt(x2**2+y2**2+z2**2)
        lx = x2/Le
        ly = y2/Le
        lz = z2/Le
        # stiffness matrix for the element
        T = np.array([[lx**2, lx*ly, lx*lz], [lx*ly, ly**2, ly*lz], [lx*lz, ly*lz, lz**2]])
        Ke = (E*A[e])/Le * np.asarray(np.bmat([[T, -T], [-T, T]]))

        # computation of the global stiffness matrix
        elementDof = np.array([[indice[0]*3, indice[0]*3+1, indice[0]*3+2, indice[1]*3, indice[1]*3+1, indice[1]*3+2]])
        stiffness[elementDof, elementDof.T] = stiffness[elementDof, elementDof.T] + Ke

    return stiffness

def solution(GDof, prescribedDof, stiffness, force):
    activeDof = np.setdiff1d(np.arange(0, GDof), prescribedDof)

    U = np.linalg.solve(stiffness[activeDof, activeDof[:, None]], force[activeDof])
    displacements = np.zeros([GDof])
    displacements[activeDof] = U
    return displacements

def stresses3Dtruss(numberElements, elementNodes, xx, yy, zz, displacements, E):
    sigma = np.zeros([numberElements])
    for e in range(0, numberElements):
        indice = elementNodes[e, :]
        elementDof = np.array([[indice[0]*3, indice[0]*3+1, indice[0]*3+2, indice[1]*3, indice[1]*3+1, indice[1]*3+2]])
        # calcul x2-x1 et y2-y1
        x2 = xx[indice[1]]-xx[indice[0]]
        y2 = yy[indice[1]] - yy[indice[0]]
        z2 = zz[indice[1]] - zz[indice[0]]
        # length of element and cos/sin angle
        Le = np.sqrt(x2**2+y2**2+z2**2)
        lx = x2 / Le
        ly = y2 / Le
        lz = z2 / Le
        # Stress
        sigma[e] = E/Le * np.array([[-lx, -ly, -lz, lx, ly, lz]])@displacements[elementDof].T

    print('Stress in elements')
    print(sigma[:, None])


def outputDisplacementsReactions(displacements, stiffness, GDof, prescribedDof):
    # displacements
    print('Displacements')
    jj = np.linspace(0, GDof - 1, GDof).reshape(GDof, 1)
    print(np.append(jj, displacements[:, None], axis=1))

    # reactions
    F = np.matmul(stiffness, displacements)
    reactions = F[prescribedDof]
    print('Reactions')
    print(np.append(prescribedDof[:, None], reactions[:, None], axis=1))

E = 1.2e6
A = np.array([0.3, 0.1, 0.2])
F = 1000

# coordinates and connectivities
numberElements = 3
numberNodes = 4
elementNodes = np.array([[0, 1], [0, 2], [0, 3]])
nodeCoor = np.array([[30, 0, 0], [0, 10, 0], [0, 10, 25], [0, 0, -15]])
xx = nodeCoor[:, 0]
yy = nodeCoor[:, 1]
zz = nodeCoor[:, 2]

# for structure
GDof = 3*numberNodes
displacements = np.zeros([GDof])
force = np.zeros([GDof])
force[0] = F

# computation of the system stiffness matrix
stiffness = formStiffness3Dtruss(GDof, numberElements, elementNodes, xx, yy, zz, E, A)

# boundary conditions and solution
prescribedDof = np.arange(3, GDof)
displacements = solution(GDof, prescribedDof, stiffness, force)

# output displacements/reactions
outputDisplacementsReactions(displacements, stiffness, GDof, prescribedDof)

# stresses at elements
stresses3Dtruss(numberElements, elementNodes, xx, yy, zz, displacements, E)

# deformed structure
scaling = 1
new_xx = xx + scaling*displacements[np.arange(0, GDof, 3)]
new_yy = yy + scaling*displacements[np.arange(1, GDof, 3)]
new_zz = zz + scaling*displacements[np.arange(2, GDof, 3)]

vec_F = np.append(nodeCoor[0, :], np.array([10*scaling*displacements[0], 0, 0]))

plt.figure()
ax = plt.axes(projection='3d')
plt.grid()
for e in range(0,numberElements):
    # initial structure
    indice = elementNodes[e, :]
    x_plot = np.array([xx[indice[0]], xx[indice[1]]])
    y_plot = np.array([yy[indice[0]], yy[indice[1]]])
    z_plot = np.array([zz[indice[0]], zz[indice[1]]])
    ax.plot3D(x_plot, y_plot, z_plot, '-ok', lw=1)

    # deformed structure
    x2_plot = np.array([new_xx[indice[0]], new_xx[indice[1]]])
    y2_plot = np.array([new_yy[indice[0]], new_yy[indice[1]]])
    z2_plot = np.array([new_zz[indice[0]], new_zz[indice[1]]])
    ax.plot3D(x2_plot, y2_plot, z2_plot, '--ok', lw=1)

ax.quiver(vec_F[0], vec_F[1], vec_F[2], vec_F[3], vec_F[4], vec_F[5], color='r', lw=2)

plt.title('Deformation of the truss')
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('z')
plt.legend(['Initial struture', 'Deformed structure (scaled)'])
plt.show()

import quadpy
import matplotlib.pyplot as plt
import csv
import math
import numpy as np
from scipy.integrate import simpson
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator


def get_point_cloud():
    pcld = np.random.random((1000, 3))
    return pcld


def delaunay(pcld):
    tri = Delaunay(pcld)
    return tri


def interpolate(tri, v, p):
    interp = LinearNDInterpolator(tri, v)
    val = interp(p)
    return val


def f(x):
    return x ** 2


def quadrature(v):
    v = v[~np.isnan(v)]
    return simpson(v)


def plot_3d(tri, pcld):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(pcld[:, 0], pcld[:, 1], pcld[:, 2], triangles=tri.simplices, cmap=plt.cm.Spectral)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


if __name__ == '__main__':
    print("This program takes points and values from the user, creates a triangulation,"
          "\nand interpolates for a specified amount of points.")

    # Training values
    values = np.random.random(1000)
    points_to_interpolate = np.random.random((1000, 3))

    points = get_point_cloud()
    delaunay = delaunay(points)
    interpolated_values = interpolate(delaunay, values, points_to_interpolate)
    print(quadrature(interpolated_values))
    plot_3d(delaunay, points)

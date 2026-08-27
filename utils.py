# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

def cubeplot3d(ax,
               phi,
               L=1,
               cmap="RdBu",
               manual_index=False,
               rcount=64,
               rasterized=True,
               visible_only=False,
               show_edges=True,
               edge_style={}):
    """Plot a 3D scalar field on the six surfaces of a cube.

    Parameters
    ----------
    ax : Axes3D
        Matplotlib axes to use for plotting.
    phi : 3D NumPy array
        Array containing the scalar field.
    L : float, optional
        Side length of the cube.
    cmap : string, optional
        Color map to use.
    manual_index : bool, optional
        NumPy's meshgrid uses a Cartesian indexing scheme.  Setting this
        to true interchanges the axes to correspond to the manual index
        ordering used in fftw_complex arrays.  This isn't strictly
        required -- all it does is to rotate the resulting cube.
    rcount : int, optional
        Maximum number of samples used in each direction.  See
        documentation of Axes3D.plot_wireframe().
    rasterized : bool, optional
        Plot faces as raster graphics
    visible_only : bool, optional
        Only plot those faces that are usually visible in Matplotlib's
        default 3D view.
    show_edges : bool, optional
        When `visible_only' is true, draw a wireframe around the cube.
    edge_style : dictionary, optional
        Keyword arguments to Axes.plot()

    Returns
    -------
    ax : Axes3D
        The axes after plotting.
    """
    phi = np.asarray(phi)
    n = phi.shape[0]
    if phi.shape != (n, n, n):
        raise ValueError("The scalar field `phi' is not in the form of a 3D cube.")

    if manual_index:
        # The axes permutation was found by trial and error.
        phi = np.transpose(phi, [1, 2, 0])

    # Meshgrid for each face.
    x = np.linspace(-L, L, n)
    p, q = np.meshgrid(x, x)

    norm = Normalize(vmin=phi.min(), vmax=phi.max())
    colors = lambda x: plt.get_cmap(cmap)(norm(x))
    kwargs = {"rcount": rcount, "ccount": rcount, "shade": False, "rasterized": rasterized}

    # Below, the normals refer to the positive x, y, or z axes.
    # A face is considered "top" or "bottom" according to the positive direction.
    top, bottom = np.full_like(p, L), np.full_like(p, -L)

    # normal = x; face = top
    phi_face = phi[:, -1, :]
    ax.plot_surface(top, q, p, facecolors=colors(phi_face), **kwargs)
    # normal = y; face = bottom
    phi_face = phi[0, :, :]
    ax.plot_surface(q, bottom, p, facecolors=colors(phi_face), **kwargs)
    # normal = z; face = top
    phi_face = phi[:, :, -1]
    ax.plot_surface(p, q, top, facecolors=colors(phi_face), **kwargs)

    if not visible_only:
        # normal = x; face = bottom
        phi_face = phi[:, 0, :]
        ax.plot_surface(bottom, q, p, facecolors=colors(phi_face), **kwargs)
        # normal = y; face = top
        phi_face = phi[-1, :, :]
        ax.plot_surface(q, top, p, facecolors=colors(phi_face), **kwargs)
        # normal = z; face = bottom
        phi_face = phi[:, :, 0]
        ax.plot_surface(p, q, bottom, facecolors=colors(phi_face), **kwargs)

    if visible_only and show_edges:
        kwargs_edges = {"zorder": 100, "color": "black", "linestyle": "solid"}
        kwargs_edges.update(edge_style)
        ax.plot([-L, -L, L, L, -L], [-L, -L, -L, -L, -L], [-L, L, L, -L, -L], **kwargs_edges)
        ax.plot([-L, L, L], [L, L, L], [L, L, -L], **kwargs_edges)
        ax.plot([-L, -L], [-L, L], [L, L], **kwargs_edges)
        ax.plot([L, L], [-L, L], [L, L], **kwargs_edges)
        ax.plot([L, L], [-L, L], [-L, -L], **kwargs_edges)

    return ax

def rescale(a, interval=(-1, 1)):
    """Rescale the values of the given array into a desired interval."""
    return (interval[0] + (a - np.min(a)) * (interval[1] - interval[0]) / (np.max(a) - np.min(a)))

def resize(psi, size=None):
    if size:
        psi = cv2.resize(psi, dsize=(size, size), interpolation=cv2.INTER_LINEAR)
    return psi

def gprange(start, end, num=100):
    """Return a geometric progression between start and end.

    Returns a geometric progression between start and end (inclusive).

    Parameters
    ----------
    start : float
        Starting point of the progression.
    end : float
        Ending point of the progression.
    num : int, optional (default = 100)
        Number of points between start and end (inclusive).

    Returns
    -------
    gp : array
        Required geometric progression.
    """
    if end / start > 0:
        ratio = (end / start)**(1.0 / (num-1))
    elif end / start < 0 and num % 2 == 0:
        ratio = -abs(end / start)**(1.0 / (num-1))
    else:
        raise ValueError('If start and end have different signs, '
                         'a real ratio is possible iff num is even.')

    return start * ratio**np.arange(num)

def imread(name, size=None, normalize=True):
    """Read an image and average RGB values to produce a grayscale image."""
    im = cv2.imread(name)

    # Linearly resize image as a square of side size.
    if size:
        im = cv2.resize(im, dsize=(size, size), interpolation=cv2.INTER_LINEAR)

    if len(im.shape) > 2:
        im = np.sum(im, axis=2) / 3

    # Normalize the image such that all values are between 0 and 1.
    if normalize:
        return (im - np.min(im)) / (np.max(im) - np.min(im))
    else:
        return im

def minmax(a):
    """Return the maximum - minimum of an array."""
    a_x = np.mean(a, axis=1)
    a_y = np.mean(a, axis=0)
    return a_x.min(), a_x.max(), a_y.min(), a_y.max()

# -*- coding: utf-8 -*-
# The MIT License (MIT)

# Copyright (c) 2016-2019 Nico Schlömer

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Code Borrowed from [meshzoo](https://github.com/nschloe/meshzoo)

import numpy


# pylint: disable=too-many-locals, too-many-statements
def _refine(node_coords, cells_nodes, edge_nodes, cells_edges):
    """Canonically refine a mesh by inserting nodes at all edge midpoints
    and make four triangular elements where there was one.
    This is a very crude refinement; don't use for actual applications.
    """
    num_nodes = len(node_coords)
    num_new_nodes = len(edge_nodes)

    # new_nodes = numpy.empty(num_new_nodes, dtype=numpy.dtype((float, 2)))
    node_coords.resize(num_nodes + num_new_nodes, 3, refcheck=False)
    # Set starting index for new nodes.
    new_node_gid = num_nodes

    # After the refinement step, all previous edge-node associations will be
    # obsolete, so record *all* the new edges.
    num_edges = len(edge_nodes)
    num_cells = len(cells_nodes)
    assert num_cells == len(cells_edges)
    num_new_edges = 2 * num_edges + 3 * num_cells
    new_edges_nodes = numpy.empty(num_new_edges, dtype=numpy.dtype((int, 2)))

    new_edge_gid = 0

    # After the refinement step, all previous cell-node associations will be
    # obsolete, so record *all* the new cells.
    num_new_cells = 4 * num_cells
    new_cells_nodes = numpy.empty(num_new_cells, dtype=numpy.dtype((int, 3)))
    new_cells_edges = numpy.empty(num_new_cells, dtype=numpy.dtype((int, 3)))
    new_cell_gid = 0

    is_edge_divided = numpy.zeros(num_edges, dtype=bool)
    edge_midpoint_gids = numpy.empty(num_edges, dtype=int)
    edge_newedges_gids = numpy.empty(num_edges, dtype=numpy.dtype((int, 2)))

    # Loop over all elements.
    for cell_id, cell in enumerate(zip(cells_edges, cells_nodes)):
        cell_edges, cell_nodes = cell
        # Divide edges.
        local_edge_midpoint_gids = numpy.empty(3, dtype=int)
        local_edge_newedges = numpy.empty(3, dtype=numpy.dtype((int, 2)))
        local_neighbor_midpoints = [[], [], []]
        local_neighbor_newedges = [[], [], []]
        for k, edge_gid in enumerate(cell_edges):
            edgenodes_gids = edge_nodes[edge_gid]
            if is_edge_divided[edge_gid]:
                # Edge is already divided. Just keep records for the cell
                # creation.
                local_edge_midpoint_gids[k] = edge_midpoint_gids[edge_gid]
                local_edge_newedges[k] = edge_newedges_gids[edge_gid]
            else:
                # Create new node at the edge midpoint.
                node_coords[new_node_gid] = 0.5 * (
                    node_coords[edgenodes_gids[0]] + node_coords[edgenodes_gids[1]]
                )
                local_edge_midpoint_gids[k] = new_node_gid
                new_node_gid += 1
                edge_midpoint_gids[edge_gid] = local_edge_midpoint_gids[k]

                # Divide edge into two.
                new_edges_nodes[new_edge_gid] = numpy.array(
                    [edgenodes_gids[0], local_edge_midpoint_gids[k]]
                )
                new_edge_gid += 1
                new_edges_nodes[new_edge_gid] = numpy.array(
                    [local_edge_midpoint_gids[k], edgenodes_gids[1]]
                )
                new_edge_gid += 1

                local_edge_newedges[k] = [new_edge_gid - 2, new_edge_gid - 1]
                edge_newedges_gids[edge_gid] = local_edge_newedges[k]
                # Do the household.
                is_edge_divided[edge_gid] = True
            # Keep a record of the new neighbors of the old nodes.
            # Get local node IDs.
            edgenodes_lids = [
                numpy.nonzero(cell_nodes == edgenodes_gids[0])[0][0],
                numpy.nonzero(cell_nodes == edgenodes_gids[1])[0][0],
            ]
            local_neighbor_midpoints[edgenodes_lids[0]].append(
                local_edge_midpoint_gids[k]
            )
            local_neighbor_midpoints[edgenodes_lids[1]].append(
                local_edge_midpoint_gids[k]
            )
            local_neighbor_newedges[edgenodes_lids[0]].append(local_edge_newedges[k][0])
            local_neighbor_newedges[edgenodes_lids[1]].append(local_edge_newedges[k][1])

        new_edge_opposite_of_local_node = numpy.empty(3, dtype=int)
        # New edges: Connect the three midpoints.
        for k in range(3):
            new_edges_nodes[new_edge_gid] = local_neighbor_midpoints[k]
            new_edge_opposite_of_local_node[k] = new_edge_gid
            new_edge_gid += 1

        # Create new elements.
        # Center cell:
        new_cells_nodes[new_cell_gid] = local_edge_midpoint_gids
        new_cells_edges[new_cell_gid] = new_edge_opposite_of_local_node
        new_cell_gid += 1
        # The three corner elements:
        for k in range(3):
            new_cells_nodes[new_cell_gid] = numpy.array(
                [
                    cells_nodes[cell_id][k],
                    local_neighbor_midpoints[k][0],
                    local_neighbor_midpoints[k][1],
                ]
            )
            new_cells_edges[new_cell_gid] = numpy.array(
                [
                    new_edge_opposite_of_local_node[k],
                    local_neighbor_newedges[k][0],
                    local_neighbor_newedges[k][1],
                ]
            )
            new_cell_gid += 1

    return node_coords, new_cells_nodes, new_edges_nodes, new_cells_edges


def create_edges(cells_nodes):
    """Setup edge-node and edge-cell relations. Adapted from voropy.
    """
    # Create the idx_hierarchy (nodes->edges->cells), i.e., the value of
    # `self.idx_hierarchy[0, 2, 27]` is the index of the node of cell 27, edge
    # 2, node 0. The shape of `self.idx_hierarchy` is `(2, 3, n)`, where `n` is
    # the number of cells. Make sure that the k-th edge is opposite of the k-th
    # point in the triangle.
    local_idx = numpy.array([[1, 2], [2, 0], [0, 1]]).T
    # Map idx back to the nodes. This is useful if quantities which are in
    # idx shape need to be added up into nodes (e.g., equation system rhs).
    nds = cells_nodes.T
    idx_hierarchy = nds[local_idx]

    s = idx_hierarchy.shape
    a = numpy.sort(idx_hierarchy.reshape(s[0], s[1] * s[2]).T)

    b = numpy.ascontiguousarray(a).view(
        numpy.dtype((numpy.void, a.dtype.itemsize * a.shape[1]))
    )
    _, idx, inv, cts = numpy.unique(
        b, return_index=True, return_inverse=True, return_counts=True
    )

    # No edge has more than 2 cells. This assertion fails, for example, if
    # cells are listed twice.
    assert all(cts < 3)

    edge_nodes = a[idx]
    cells_edges = inv.reshape(3, -1).T

    return edge_nodes, cells_edges


def show2d(*args, **kwargs):
    import matplotlib.pyplot as plt

    plot2d(*args, **kwargs)
    plt.show()
    return


def plot2d(points, cells, mesh_color="k", show_axes=False):
    """Plot a 2D mesh using matplotlib.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig = plt.figure()
    ax = fig.gca()
    plt.axis("equal")
    if not show_axes:
        ax.set_axis_off()

    xmin = numpy.amin(points[:, 0])
    xmax = numpy.amax(points[:, 0])
    ymin = numpy.amin(points[:, 1])
    ymax = numpy.amax(points[:, 1])

    width = xmax - xmin
    xmin -= 0.1 * width
    xmax += 0.1 * width

    height = ymax - ymin
    ymin -= 0.1 * height
    ymax += 0.1 * height

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    edge_nodes, _ = create_edges(cells)

    # Get edges, cut off z-component.
    e = points[edge_nodes][:, :, :2]
    line_segments = LineCollection(e, color=mesh_color)
    ax.add_collection(line_segments)
    return fig


def iso_sphere(ref_steps=4):
    # Start off with an isosahedron and refine.

    # Construction from
    # <http://blog.andreaskahler.com/2009/06/creating-icosphere-mesh-in-code.html>.
    # Create 12 vertices of a icosahedron.
    t = (1.0 + numpy.sqrt(5.0)) / 2.0
    nodes = numpy.array(
        [
            [-1, +t, +0],
            [+1, +t, +0],
            [-1, -t, +0],
            [+1, -t, +0],
            #
            [+0, -1, +t],
            [+0, +1, +t],
            [+0, -1, -t],
            [+0, +1, -t],
            #
            [+t, +0, -1],
            [+t, +0, +1],
            [-t, +0, -1],
            [-t, +0, +1],
        ]
    )

    cells_nodes = numpy.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ]
    )

    # Refine.
    edge_nodes, cells_edges = create_edges(cells_nodes)
    args = nodes, cells_nodes, edge_nodes, cells_edges
    for _ in range(ref_steps):
        args = _refine(*args)

    # push all nodes to the sphere
    nodes = args[0]
    nodes = (nodes.T / numpy.sqrt(numpy.einsum("ij,ij->i", nodes, nodes)).T).T

    return nodes, args[1]
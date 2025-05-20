from collections.abc import Sequence

import geopandas as gpd
import numpy as np
import torch
from matplotlib import pyplot as plt
import rasterio
from rasterio.plot import show
from shapely.geometry import LinearRing, MultiPolygon, Polygon, box
from shapely.ops import unary_union


class AoiSampler:
    """Allows efficiently sampling points inside an AOI uniformly at random.

    To achieve this, each polygon in the AOI is first partitioned into
    triangles (triangulation). Then, to sample a single point, we first sample
    a triangle at random with probability proportional to its area and then
    sample a point within that triangle uniformly at random.

    This is taken from rastervision AioSampler:
    https://github.com/azavea/raster-vision/blob/master/rastervision_core/rastervision/core/data/utils/aoi_sampler.py
    """

    def __init__(
        self, polygons: Sequence[Polygon], roi_box=None, size_lims=(128, 258)
    ) -> None:
        """Args:
        polygons: List of shapely Polygon object.
        roi: Region of interest. Defaults to None.
        """
        self.size_lims = size_lims
        # merge overlapping polygons, if any
        merged_polygons = unary_union(polygons)
        if roi_box is not None:
            merged_polygons = merged_polygons.intersection(roi_box)

        if isinstance(merged_polygons, Polygon):
            merged_polygons = [merged_polygons]
        elif isinstance(merged_polygons, MultiPolygon):
            merged_polygons = list(merged_polygons.geoms)
        self.polygons = merged_polygons
        self.triangulate(self.polygons)

    def triangulate(self, polygons) -> dict:
        triangulations = [self.triangulate_polygon(p) for p in polygons]
        self.triangulations = triangulations
        self.origins = np.vstack([t["origins"] for t in triangulations])
        self.vec_AB = np.vstack([t["bases"][0] for t in triangulations])
        self.vec_AC = np.vstack([t["bases"][1] for t in triangulations])
        areas = np.concatenate([t["areas"] for t in triangulations])
        self.weights = areas / areas.sum()
        self.ntriangles = len(self.origins)

    def sample(self, n: int = 1) -> np.ndarray:
        """Sample a random point within the AOI, using the following algorithm:
            - Randomly sample one triangle (ABC) with probability proportional
            to its area.
            - Starting at A, travel a random distance along vectors AB and AC.
            - Return the final position.

        Args:
            n (int): Number of points to sample. Defaults to 1.

        Returns:
            np.ndarray: (n, 2) 2D coordinates of the sampled points.
        """
        tri_idx = np.random.choice(self.ntriangles, p=self.weights, size=n)
        origin = self.origins[tri_idx]
        vec_AB = self.vec_AB[tri_idx]
        vec_AC = self.vec_AC[tri_idx]
        # the fractions to travel along each of the two vectors
        r, s = np.random.uniform(size=(2, n, 1))
        # If the fractions will land us in the wrong half of the parallelogram
        # defined by vec AB and vec AC, reflect them into the correct half.
        mask = (r + s) > 1
        r[mask] = 1 - r[mask]
        s[mask] = 1 - s[mask]
        loc = origin + (r * vec_AB + s * vec_AC)
        return loc

    def sample_grid(
        self,
        window_size_scaled: int,
        overlap: float = 0.0,
        polygon_intersection: float = 0.5,
    ) -> list[Polygon]:
        """Sample windows in a grid pattern covering the entire polygon region.

        Works by traversing the grid with a stride length of `window_size_scaled * (1 - overlap)`
        and sampling only windows that have at least `polygon_intersection` percentage of overlap with the given polygons.

        Args:
            window_size_scaled: Size of each window (assumed square) - scaled by the the raster resolution
            overlap: Overlap between adjacent windows (0.0 to 1.0)
            polygon_intersection: Minimum intersection percentage with the given polygons. (e.g. training areas/polygons)

        Returns:
            List of window polygons
        """
        # Get bounds of all polygons
        bounds = unary_union(self.polygons).bounds
        minx, miny, maxx, maxy = bounds

        # Calculate step size based on overlap
        step = window_size_scaled * (1 - overlap)

        windows = []
        y = miny
        while y < maxy:
            x = minx
            while x < maxx:
                window = box(x, y, x + window_size_scaled, y + window_size_scaled)
                # Only keep windows that have at least 50% intersection with polygons
                window_area = window.area
                for polygon in self.polygons:
                    if (
                        window.intersection(polygon).area / window_area
                        >= polygon_intersection
                    ):
                        windows.append(window)
                        break
                x += step
            y += step

        return windows

    def triangulate_polygon(self, polygon: Polygon) -> dict:
        """Triangulate polygon.

        Extracts vertices and edges from the polygon (and its holes, if any)
        and passes them to the Triangle library for triangulation.
        """
        from triangle import triangulate

        vertices, edges = self.polygon_to_graph(polygon)

        holes = polygon.interiors
        if not holes:
            args = {
                "vertices": vertices,
                "segments": edges,
            }
        else:
            for hole in holes:
                hole_vertices, hole_edges = self.polygon_to_graph(hole)
                # make the indices point to entries in the global vertex list
                hole_edges += len(vertices)
                # append hole vertices to the global vertex list
                vertices = np.vstack([vertices, hole_vertices])
                edges = np.vstack([edges, hole_edges])

            # the triangulation algorithm requires a sample point inside each
            # hole
            hole_centroids = [hole.centroid for hole in holes]
            hole_centroids = np.concatenate(
                [np.array(c.coords) for c in hole_centroids], axis=0
            )

            args = {"vertices": vertices, "segments": edges, "holes": hole_centroids}

        tri = triangulate(args, opts="p")
        simplices = tri["triangles"]
        vertices = np.array(tri["vertices"])
        origins, bases = self.triangle_origin_and_basis(vertices, simplices)

        out = {
            "vertices": vertices,
            "simplices": simplices,
            "origins": origins,
            "bases": bases,
            "areas": self.triangle_area(vertices, simplices),
        }
        return out

    def polygon_to_graph(
        self, polygon: Polygon | LinearRing
    ) -> tuple[np.ndarray, np.ndarray]:
        """Given a polygon, return its graph representation.

        Args:
            polygon: A polygon or polygon-exterior.

        Returns:
            An (N, 2) array of vertices and an (N, 2) array of indices to
            vertices representing edges.
        """
        exterior = getattr(polygon, "exterior", polygon)
        vertices = np.array(exterior.coords)
        # Discard the last vertex - it is a duplicate of the first vertex and
        # duplicates cause problems for the Triangle library.
        vertices = vertices[:-1]

        N = len(vertices)
        # Tuples of indices to vertices representing edges.
        # mod N ensures edge from last vertex to first vertex by making the
        # last tuple [N-1, 0].
        edges = np.column_stack([np.arange(0, N), np.arange(1, N + 1)]) % N

        return vertices, edges

    def triangle_side_lengths(
        self, vertices: np.ndarray, simplices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate lengths of all 3 sides of each triangle specified by the
        simplices array.

        Args:
            vertices: (N, 2) array of vertex coords in 2D.
            simplices: (N, 3) array of indexes to entries in the
                vertices array. Each row represents one triangle.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: ||AB||, ||BC||, ||AC||
        """
        A = vertices[simplices[:, 0]]
        B = vertices[simplices[:, 1]]
        C = vertices[simplices[:, 2]]
        AB, AC, BC = B - A, C - A, C - B
        ab = np.linalg.norm(AB, axis=1)
        bc = np.linalg.norm(BC, axis=1)
        ac = np.linalg.norm(AC, axis=1)
        return ab, bc, ac

    def triangle_origin_and_basis(
        self, vertices: np.ndarray, simplices: np.ndarray
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """For each triangle ABC, return point A, vector AB, and vector AC.

        Args:
            vertices: (N, 2) array of vertex coords in 2D.
            simplices: (N, 3) array of indexes to entries in the
                vertices array. Each row represents one triangle.

        Returns:
            3 arrays of shape (N, 2), organized into tuples like so:
            (point A, (vector AB, vector AC)).
        """
        A = vertices[simplices[:, 0]]
        B = vertices[simplices[:, 1]]
        C = vertices[simplices[:, 2]]
        AB = B - A
        AC = C - A
        return A, (AB, AC)

    def triangle_area(self, vertices: np.ndarray, simplices: np.ndarray) -> np.ndarray:
        """Calculate area of each triangle specified by the simplices array
        using Heron's formula.

        Args:
            vertices: (N, 2) array of vertex coords in 2D.
            simplices: (N, 3) array of indexes to entries in the
                vertices array. Each row represents one triangle.

        Returns:
            np.ndarray: (N,) array of areas
        """
        a, b, c = self.triangle_side_lengths(vertices, simplices)
        p = (a + b + c) * 0.5
        area = p * (p - a) * (p - b) * (p - c)
        area[area < 0] = 0
        area = np.sqrt(area)
        return area

    def sample_window_size(self) -> tuple[int, int]:
        """Randomly sample the window size."""
        sz_min, sz_max = self.size_lims
        if sz_max == sz_min + 1:
            return sz_min, sz_min
        # randomly sample windows given (float) size minimum and maximum
        size = torch.rand(1) * (sz_max - sz_min) + sz_min
        return size.item(), size.item()

    def sample_window_loc(self, h: int, w: int) -> tuple[int, int]:
        """Randomly sample coordinates of the top left corner of the window."""
        x, y = self.sample().round().T
        x, y = int(x.item()), int(y.item())
        return x, y

    def sample_window(self, polygon_intersection: int = 0.0):
        """Randomly sample a window that satisfies the polygon_intersection (0-1), that is
        minimum intersection percentage with the given polygons.
        """
        h, w = self.sample_window_size()
        x, y = self.sample_window_loc(h, w)
        window = box(x, y, x + w, y + h)
        intersection_area = sum(
            window.intersection(polygon).area for polygon in self.polygons
        )
        intersection_percentage = (intersection_area / window.area)

        if intersection_percentage <= polygon_intersection:
            return self.sample_window(polygon_intersection)
        else:
            return window

    def show_windows(
        self,
        polygons,
        windows,
        image=None,
        boundary_shape=None,
        raster_transform=None,
        title="Sampled Windows",
    ) -> None:
        """Visualize generated windows along with the raster image and fenced area if given
        Args:
            polygons: a list of grids in polygons
            windows: windows to visualize
            image: raster image
            boundary_shape: boundary of the raster image (fenced area), it should be
                polygon type
            raster_transform: raster transform
            title: title of the plot.
        """
        fig, ax = plt.subplots()

        if image is not None:
            with rasterio.open(image) as src:
                show(src, ax=ax)

        for polygon in polygons:
            x, y = polygon.exterior.xy
            ax.fill(x, y, alpha=0.5, fc="gray", edgecolor="black")

        # draw windows on top of the image
        for w in windows:
            x, y = w.exterior.xy
            ax.plot(x, y, color="red")

        # plot boudary shape (fenced_area)
        if boundary_shape:
            gpd.GeoSeries(boundary_shape).boundary.plot(
                ax=ax, color="green", linewidth=2
            )

        ax.autoscale()
        ax.set_title(title)
        plt.show()

        return fig
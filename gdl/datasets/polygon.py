import matplotlib.pyplot as plt
import rasterio
from matplotlib.patches import Polygon
from rasterio.plot import show

from .utils import create_intersecting_grids


class PolygonSplitter:
    """A class helper for spliting the given polygon into train, validation and test polygons."""

    def __init__(self, grid_path, fenced_area_path) -> None:
        self.grid_path = grid_path
        self.fenced_area_path = fenced_area_path
        self.intersecting_grids = create_intersecting_grids(
            self.grid_path, self.fenced_area_path
        )

    def get_polygon_by_indices(self, grid_indices):
        """Create train polygon given indices of the grid cells.

        Args:
            grid_indices (list): List of grid indices.

        Returns:
            list[shapely.geometry.Polygon]: List of polygons.
        """
        polygons = []
        for _, grid in self.intersecting_grids.iterrows():
            if grid["id"] in grid_indices:
                polygons.append(grid["geometry"])
        return polygons

    def plot(self, raster_image, polygons, ax=None, title="Polygon visualization") -> None:
        """Plot the raster image with the given polygons.

        Args:
            raster_image (str): Path to the raster image.
            polygons (list): List of polygons.
        """
        if ax is None:
            fig, ax = plt.subplots()

        with rasterio.open(raster_image) as src:
            show(src, ax=ax)

            for polygon in polygons:
                exterior_coords = polygon.exterior.coords
                poly_patch = Polygon(
                    exterior_coords, edgecolor="red", lw=2, facecolor="none"
                )
                ax.add_patch(poly_patch)
        ax.set_title(title)
        plt.show()

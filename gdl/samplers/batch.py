from collections.abc import Iterator

import torch
from torchgeo.samplers import BatchGeoSampler
from torchgeo.datasets import GeoDataset, BoundingBox

from torchgeo.samplers.constants import Units
from torchgeo.samplers.utils import _to_tuple

from gdl.samplers.aio_sampler import AoiSampler
from shapely.geometry import MultiPolygon, Polygon, box
from typing import Optional


class RandomBatchAoiGeoSampler(BatchGeoSampler):
    """Samples elements from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random chips/images as possible. Note that
    randomly sampled chips may overlap.

    ToDo: add max retries
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size_lims: tuple[float, float],
        polygons: list[Polygon],
        length: Optional[int],
        batch_size: int,
        intersection_percentage_th: float = 75.0,
        roi: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
        max_retries: int = 50000,
        outer_boundary_shape: str = None,
    ) -> None:
        """Initialize a new Sampler instance.
        Args:
            dataset: dataset to index from
            size_lims: minimum and maximum size limit to sample windows
            polygons: list of polygons in which the windows will be sampled from
            length: number of random samples to draw per epoch
                (defaults to approximately the maximal number of non-overlapping
                :term:`chips <chip>` of size ``size`` that could be sampled from
                the dataset)
            batch_size: number of batch size
            intersection_percentage_th: percentage of the intersection of sampled windows
                with the union of polygons. Set the percentage to 100 if you want to 
                sample windows from the polygons only. The higher percentage may take 
                longer to find random windows within the polygons.
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size_limit`` is in pixel or CRS units
            max_retries: ToDo: maximum retries to find windows inside the polygons, because
                it may end up in continuous loop
            outer_boundary_shape: path to the shapefile that defines the outer boundary of the field
                e.g. fenced area shape
        """
        super().__init__(dataset, roi)

        self.size_lims = _to_tuple(size_lims)

        if units == Units.PIXELS:
            self.size_lims = (self.size_lims[0] * self.res, self.size_lims[1] * self.res)

        # create shapey box for the dataset roi
        if outer_boundary_shape is not None:
            outer_boundary_shape = gpd.read_file(outer_boundary_shape)
            self.outer_shape = outer_boundary_shape.geometry.union_all()
        else:
            self.outer_shape = box(self.roi.minx, self.roi.miny, self.roi.maxx, self.roi.maxy)

        # make sure that both aoi_sampler and multi_polygons are within the roi_box
        self.aoi_sampler = AoiSampler(polygons, self.outer_shape)
        self.multi_polygons = MultiPolygon(polygons).intersection(self.outer_shape)
        if isinstance(self.multi_polygons, Polygon):
            self.multi_polygons = [self.multi_polygons]
        elif isinstance(self.multi_polygons, MultiPolygon):
            self.multi_polygons = list(self.multi_polygons.geoms)

        self.length = length
        self.batch_size = batch_size

        # ToDo: change the size to pixel units

    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        for _ in range(len(self)):
            windows = [
                self.sample_window(intersection_percentage_th=90) for _ in range(self.batch_size)
            ]
            batch_windows = []
            for window in windows:
                min_x, min_y, max_x, max_y = window.bounds
                bbox = BoundingBox(
                    min_x, max_x, min_y, max_y, self.roi.mint, self.roi.maxt
                )
                batch_windows.append(bbox)

            yield batch_windows

    def __len__(self) -> int:
        """Return the number of samples in a single epoch.

        Returns:
            length of the epoch
        """
        return self.length

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
        x, y = self.aoi_sampler.sample().round().T
        x, y = int(x.item()), int(y.item())
        return x, y

    def sample_window(self, intersection_percentage_th=0.0):
        """Randomly sample a window that satisfies the intersection_percentage_th, that is
        minimum intersection percentage with the given polygons.
        """
        h, w = self.sample_window_size()
        x, y = self.sample_window_loc(h, w)
        window = box(x, y, x + w, y + h)
        intersection_area = sum(
            window.intersection(polygon).area for polygon in self.multi_polygons
        )
        intersection_percentage = (intersection_area / window.area) * 100

        if intersection_percentage <= intersection_percentage_th:
            return self.sample_window(intersection_percentage_th)
        else:
            return window


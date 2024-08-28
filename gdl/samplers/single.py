import abc
from collections.abc import Iterator

import geopandas as gpd
import torch
from rtree.index import Index, Property
from shapely.geometry import MultiPolygon, Polygon, box
from torch.utils.data import Sampler
from torchgeo.datasets import BoundingBox, GeoDataset
from torchgeo.samplers.constants import Units
from torchgeo.samplers.utils import _to_tuple

from gdl.samplers.aoi_sampler import AoiSampler


class GeoSampler(Sampler[BoundingBox], abc.ABC):
    """Abstract base class for sampling from :class:`~torchgeo.datasets.GeoDataset`.

    Unlike PyTorch's :class:`~torch.utils.data.Sampler`, :class:`GeoSampler`
    returns enough geospatial information to uniquely index any
    :class:`~torchgeo.datasets.GeoDataset`. This includes things like latitude,
    longitude, height, width, projection, coordinate system, and time.
    """

    def __init__(self, dataset: GeoDataset, roi: BoundingBox | None = None) -> None:
        """Initialize a new Sampler instance.

        Args:
            dataset: dataset to index from
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
        """
        if roi is None:
            self.index = dataset.index
            roi = BoundingBox(*self.index.bounds)
        else:
            self.index = Index(interleaved=False, properties=Property(dimension=3))
            hits = dataset.index.intersection(tuple(roi), objects=True)
            for hit in hits:
                bbox = BoundingBox(*hit.bounds) & roi
                self.index.insert(hit.id, tuple(bbox), hit.object)

        self.res = dataset.res
        self.roi = roi

    @abc.abstractmethod
    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """


class RandomAoiGeoSampler(GeoSampler):
    """Samples elements from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random chips/images as possible. Note that
    randomly sampled chips may overlap.

    ToDo: add max retries
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size_lims: tuple[float, float] | float,
        polygons: list[Polygon],
        length: int | None,
        roi: BoundingBox | None = None,
        units: Units = Units.PIXELS,
        max_retries: int = 50000,
        outer_boundary_shape: str | None = None,
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
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` is in pixel or CRS units
            exclude_nodata_samples: will ensure that samples are not outside of the
                footprint of the valid pixel. No-data regions may occur due to
                re-projection or inherit no-data regions in rasters.
            max_retries: is used when exclude_nodata_samples are True. Is a safe-guard
                for infinite loops in case the nodata-mask of the raster is wrong.
                (ToDo)
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

    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        for _ in range(len(self)):
            window = self.sample_window(intersection_percentage_th=90)
            min_x, min_y, max_x, max_y = window.bounds
            bbox = BoundingBox(min_x, max_x, min_y, max_y, self.roi.mint, self.roi.maxt)
            yield bbox

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
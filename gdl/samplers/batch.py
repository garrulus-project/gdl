from collections.abc import Iterator

import torch
from shapely.geometry import MultiPolygon, Polygon, box
from torchgeo.datasets import BoundingBox, GeoDataset
from torchgeo.samplers import BatchGeoSampler
from torchgeo.samplers.constants import Units
from torchgeo.samplers.utils import _to_tuple

from gdl.samplers.aoi_sampler import AoiSampler


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
        length: int | None,
        batch_size: int,
        intersection_percentage_th: float = 75.0,
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

        # get the intersection of the polygons with the outer boundary shape
        if outer_boundary_shape is not None:
            outer_boundary_shape = gpd.read_file(outer_boundary_shape)
            self.outer_shape = outer_boundary_shape.geometry.union_all()
        else:
            self.outer_shape = box(self.roi.minx, self.roi.miny, self.roi.maxx, self.roi.maxy)

        # make sure that both aoi_sampler and multi_polygons are within the roi_box
        self.aoi_sampler = AoiSampler(polygons, self.outer_shape, self.size_lims)
        self.multi_polygons = MultiPolygon(polygons).intersection(self.outer_shape)
        if isinstance(self.multi_polygons, Polygon):
            self.multi_polygons = [self.multi_polygons]
        elif isinstance(self.multi_polygons, MultiPolygon):
            self.multi_polygons = list(self.multi_polygons.geoms)

        self.intersection_percentage_th = intersection_percentage_th
        self.length = length
        self.batch_size = batch_size

        # create random samplers, this is only generated once and will be used
        # across all the epochs
        areas = []
        self.bboxes = []
        for _ in range(self.length):
            window = self.aoi_sampler.sample_window(intersection_percentage_th=self.intersection_percentage_th)
            bbox = BoundingBox(window.bounds[0],window.bounds[2],
                               window.bounds[1], window.bounds[3], 
                               self.roi.mint, self.roi.maxt)
            self.bboxes.append(bbox)
            areas.append(bbox.area)

        # torch.multinomial requires float probabilities > 0
        self.areas = torch.tensor(areas, dtype=torch.float)
        if torch.sum(self.areas) == 0:
            self.areas += 1

    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        for _ in range(len(self)):
            # Choose a random tile, weighted by area
            batch_indices = torch.multinomial(self.areas, self.batch_size)
            yield [self.bboxes[i] for i in batch_indices]

    def __len__(self) -> int:
        """Return the number of samples in a single epoch.

        Returns:
            length of the epoch
        """
        return self.length // self.batch_size

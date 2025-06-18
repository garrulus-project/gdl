import math
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
        polygon_intersection: float = 0.75,
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
            polygon_intersection: percentage of the intersection of sampled windows
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
            self.size_lims = (
                self.size_lims[0] * self.res,
                self.size_lims[1] * self.res,
            )

        # get the intersection of the polygons with the outer boundary shape
        if outer_boundary_shape is not None:
            outer_boundary_shape = gpd.read_file(outer_boundary_shape)
            self.outer_shape = outer_boundary_shape.geometry.union_all()
        else:
            self.outer_shape = box(
                self.roi.minx, self.roi.miny, self.roi.maxx, self.roi.maxy
            )

        # make sure that both aoi_sampler and multi_polygons are within the roi_box
        self.aoi_sampler = AoiSampler(polygons, self.outer_shape, self.size_lims)
        self.multi_polygons = MultiPolygon(polygons).intersection(self.outer_shape)
        if isinstance(self.multi_polygons, Polygon):
            self.multi_polygons = [self.multi_polygons]
        elif isinstance(self.multi_polygons, MultiPolygon):
            self.multi_polygons = list(self.multi_polygons.geoms)

        self.polygon_intersection = polygon_intersection
        self.length = length
        self.batch_size = batch_size

        # initialized windows or list of bounding boxes for the dataloader
        self.sample_windows()

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

    def sample_windows(self) -> None:
        """Resample windows over the orthomosaic and update bbox list.
        This function can be called after every epoch to increase
        randomness of the sampled data over the orthomosaic.
        """
        # reset areas and bboxes
        areas = []
        self.bboxes = []
        for _ in range(self.length):
            window = self.aoi_sampler.sample_window(
                polygon_intersection=self.polygon_intersection
            )
            bbox = BoundingBox(
                window.bounds[0],
                window.bounds[2],
                window.bounds[1],
                window.bounds[3],
                self.roi.mint,
                self.roi.maxt,
            )
            self.bboxes.append(bbox)
            areas.append(bbox.area)

        # torch.multinomial requires float probabilities > 0
        self.areas = torch.tensor(areas, dtype=torch.float)
        if torch.sum(self.areas) == 0:
            self.areas += 1


class GridBatchAoiGeoSampler(BatchGeoSampler):
    """Sample windows in grid fashion for consistent sampling.
    This can be used to sample val and test orthomosaics.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: int,
        polygons: list[Polygon],
        batch_size: int,
        polygon_intersection: float = 0.5,
        window_overlap: float = 0.25,
        roi: BoundingBox | None = None,
        units: Units = Units.PIXELS,
        outer_boundary_shape: str | None = None,
    ) -> None:
        """Initialize a new Sampler instance.

        Args:
            dataset: dataset to index from
            size: size in pixel
            polygons: list of polygons in which the windows will be sampled from
            batch_size: number of batch size
            polygon_intersection: percentage of the intersection of sampled windows
                with the union of polygons. Set the percentage to 100 if you want to
                sample windows from the polygons only. The higher percentage may take
                longer to find random windows within the polygons.
            window_overlap: window overlap with neighbouring windows
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size_limit`` is in pixel or CRS units
            outer_boundary_shape: path to the shapefile that defines the outer boundary of the field
                e.g. fenced area shape
        """
        super().__init__(dataset, roi)

        if units == Units.PIXELS:
            self.size_lims = (size * self.res, size * self.res)

        # get the intersection of the polygons with the outer boundary shape
        if outer_boundary_shape is not None:
            outer_boundary_shape = gpd.read_file(outer_boundary_shape)
            self.outer_shape = outer_boundary_shape.geometry.union_all()
        else:
            self.outer_shape = box(
                self.roi.minx, self.roi.miny, self.roi.maxx, self.roi.maxy
            )

        # make sure that both aoi_sampler and multi_polygons are within the roi_box
        self.aoi_sampler = AoiSampler(polygons, self.outer_shape)
        self.multi_polygons = MultiPolygon(polygons).intersection(self.outer_shape)
        if isinstance(self.multi_polygons, Polygon):
            self.multi_polygons = [self.multi_polygons]
        elif isinstance(self.multi_polygons, MultiPolygon):
            self.multi_polygons = list(self.multi_polygons.geoms)

        self.batch_size = batch_size
        self.scaled_size = size * self.res
        self.polygon_intersection = polygon_intersection
        self.window_overlap = window_overlap
        # sample window
        self.sampled_grid_windows = self.aoi_sampler.sample_grid(
            window_size_scaled=self.scaled_size,
            overlap=self.window_overlap,
            polygon_intersection=self.polygon_intersection,
        )
        # set maximum length
        self.length = len(self.sampled_grid_windows)

        # create random samplers, this is only generated once and will be used
        # across all the epochs
        self.bboxes = []
        for window in self.sampled_grid_windows:
            bbox = BoundingBox(
                window.bounds[0],
                window.bounds[2],
                window.bounds[1],
                window.bounds[3],
                self.roi.mint,
                self.roi.maxt,
            )
            self.bboxes.append(bbox)

    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        # Iterate sequentially in the grid order.
        for i in range(0, len(self.bboxes), self.batch_size):
            yield self.bboxes[i : i + self.batch_size]

    def __len__(self) -> int:
        """Return the number of samples in a single epoch.

        Returns:
            length of the epoch.
        """
        return math.ceil(self.length / self.batch_size)


class DistributedRandomBatchAoiGeoSampler(RandomBatchAoiGeoSampler):
    def __init__(
        self,
        dataset: GeoDataset,
        size_lims: tuple[float, float],
        polygons: list[Polygon],
        length: int | None,
        batch_size: int,
        polygon_intersection: float = 0.75,
        roi: BoundingBox | None = None,
        units: Units = Units.PIXELS,
        max_retries: int = 50000,
        outer_boundary_shape: str | None = None,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__(
            dataset,
            size_lims,
            polygons,
            length,
            batch_size,
            polygon_intersection,
            roi,
            units,
            max_retries,
            outer_boundary_shape,
        )

        if num_replicas is None or rank is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError('Distributed training not initialized')

        self.num_replicas = (
            num_replicas
            if num_replicas is not None
            else torch.distributed.get_world_size()
        )
        self.rank = rank if rank is not None else torch.distributed.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        self.total_size = math.ceil(self.length / self.num_replicas) * self.num_replicas
        self.samples_per_replica = self.total_size // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[BoundingBox]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = list(range(len(self.bboxes)))
        if self.shuffle:
            indices = torch.tensor(indices)[
                torch.randperm(len(indices), generator=g)
            ].tolist()

        # pad to make divisible
        required = self.samples_per_replica * self.num_replicas
        indices += indices[: (required - len(indices))]

        # shard or partition data
        indices = indices[self.rank :: self.num_replicas]

        for i in range(0, len(indices), self.batch_size):
            yield [self.bboxes[j] for j in indices[i : i + self.batch_size]]

    def __len__(self) -> int:
        return self.samples_per_replica // self.batch_size

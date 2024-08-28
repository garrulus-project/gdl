from typing import Any

import geopandas as gpd
import kornia.augmentation as K
import torch
from kornia.constants import DataKey, Resample
from torchgeo.datamodules import GeoDataModule
from torchgeo.datasets import BoundingBox, UnionDataset
from torchgeo.datasets.splits import roi_split
from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler
from torchgeo.samplers.utils import _to_tuple
from torchgeo.transforms import AugmentationSequential

from ..datasets.benchmark import get_field_D_grid_split
from ..datasets.geo import GarrulusSegmentationDataset
from ..datasets.polygon import PolygonSplitter
from ..samplers.batch import RandomBatchAoiGeoSampler


class GarrulusAoiDataModule(GeoDataModule):
    """LightningDataModule implementation for the Garrulus dataset.

    This module handles data loading and augmentation for the Garrulus
    dataset. It allows for splitting the data based on specified grid
    IDs.
    """

    def __init__(
        self,
        raster_image_path: str | None = None,
        mask_path: str | None = None,
        grid_shape_path: str | None = None,
        fenced_area_shape_path: str | None = None,
        batch_size: int = 64,
        size_lims: tuple[float, float] = (128,256),
        img_size: int = 224,
        length: int = 1000,
        num_workers: int = 1,
        class_set: int = 5,
        use_prior_labels: bool = False,
        prior_smoothing_constant: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        """Initialize a new GarrulusAoiDataModule instance.

        Args:
            raster_image_path (str): Path to the raster image data.
            mask_path (str): Path to the mask data.
            grid_shape_path (str): Path to the shapefile containing grid shapes.
            fenced_area_shape_path (str): Path to the shapefile defining fenced areas.
            batch_size (int): Size of each mini-batch.
            size_lims (int,int): Minimum and maximum size of sampled windows
            img_size (int): Size of the input image.
            length (int | None): Length of each training epoch.
            num_workers (int): Number of workers for parallel data loading.
            class_set (int): The high-resolution land cover class set to use (5 or 7).
            use_prior_labels (bool): Whether to use a prior over high-resolution
                                     classes instead of the labels themselves.
            prior_smoothing_constant (float): Smoothing constant added when using prior labels.
            **kwargs (Any): Additional keyword arguments passed to the base class.
        """
        self.raster_image_path = raster_image_path
        self.mask_path = mask_path
        self.grid_shape_path = grid_shape_path
        self.fenced_area_shape_path = fenced_area_shape_path
        self.use_prior_labels = use_prior_labels
        self.class_set = class_set
        self.img_size = img_size
        self.size_lims = size_lims
        self.length = length

        super().__init__(
            GarrulusSegmentationDataset,
            batch_size=batch_size,
            length=self.length,
            num_workers=num_workers,
            **kwargs,
        )

        self.train_aug = AugmentationSequential(
            K.Resize(self.img_size),
            K.Normalize(mean=torch.tensor(0), std=torch.tensor(255)),
            # K.RandomResizedCrop(_to_tuple(self.patch_size), scale=(0.6, 1.0)),
            K.RandomVerticalFlip(p=0.5),
            K.RandomHorizontalFlip(p=0.5),
            data_keys=["image", "mask"],
            extra_args={
                DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None}
            },
        )
        self.test_aug = AugmentationSequential(
            K.Resize(self.img_size),
            K.Normalize(mean=torch.tensor(0), std=torch.tensor(255)),
            data_keys=["image", "mask"],
            extra_args={
                DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None}
            },
        )
        self.valid_aug = AugmentationSequential(
            K.Resize(self.img_size),
            K.Normalize(mean=torch.tensor(0), std=torch.tensor(255)),
            data_keys=["image", "mask"],
            extra_args={
                DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None}
            },
        )

    def setup(self, stage: str) -> None:
        """Set up datasets and samplers for training, validation, testing, or prediction.

        Args:
            stage (str): One of 'fit', 'validate', 'test', or 'predict'.
        """
        self.dataset = GarrulusSegmentationDataset(
            self.raster_image_path,
            self.mask_path,
            transforms=self.train_aug,
            **self.kwargs, 
        )

        ps = PolygonSplitter(self.grid_shape_path, self.fenced_area_shape_path)

        train_indices, validation_indices, test_indices = get_field_D_grid_split()
        train_polygon = ps.get_polygon_by_indices(grid_indices=train_indices)
        validation_polygon = ps.get_polygon_by_indices(grid_indices=validation_indices)
        test_polygon = ps.get_polygon_by_indices(grid_indices=test_indices)

        if stage == "fit":
            self.train_batch_sampler = RandomBatchAoiGeoSampler(
                self.dataset,
                size_lims=self.size_lims,
                polygons=train_polygon,
                batch_size=self.batch_size,
                length=self.length,
            )

        # ToDo: use grid sample within AOI for validation and test
        if stage in ["fit", "validate"]:
            self.val_sampler = RandomBatchAoiGeoSampler(
                self.dataset,
                size_lims=self.size_lims,
                polygons=validation_polygon,
                batch_size=self.batch_size,
                length=self.length,
            )

        # # ToDo: split prediction
        if stage in ["test", "predict"]:
            self.test_sampler = RandomBatchAoiGeoSampler(
                self.dataset,
                size_lims=self.size_lims,
                polygons=test_polygon,
                batch_size=self.batch_size,
                length=self.length,
            )

            self.predict_dataset = self.test_dataset
            self.predict_sampler = self.test_sampler



class GarrulusGridDataModule(GeoDataModule):
    """LightningDataModule implementation for the Garrulus dataset.

    This module handles data loading and augmentation for the Garrulus
    dataset. It allows for splitting the data based on specified grid
    IDs.
    """

    test_grid_idx = [50, 51, 52, 53, 54, 55]
    valid_grid_idx = [57, 58, 59, 60, 61, 62]
    train_grid_idx = [
        3,
        9,
        10,
        11,
        16,
        17,
        18,
        19,
        23,
        24,
        25,
        26,
        27,
        29,
        30,
        31,
        32,
        33,
        34,
        36,
        37,
        38,
        39,
        40,
        41,
        43,
        44,
        45,
        46,
        47,
        48,
    ]
    # imagenet normalization
    _mean = torch.tensor([0.485, 0.456, 0.406])
    _std = torch.tensor([0.229, 0.224, 0.225])

    def __init__(
        self,
        raster_image_path: str | None = None,
        mask_path: str | None = None,
        grid_shape_path: str | None = None,
        fenced_area_shape_path: str | None = None,
        batch_size: int = 64,
        patch_size: int = 256,
        length: int | None = None,
        num_workers: int = 1,
        class_set: int = 5,
        use_prior_labels: bool = False,
        prior_smoothing_constant: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        """Initialize a new GarrulusGridDataModule instance.

        Args:
            raster_image_path (str): Path to the raster image data.
            mask_path (str): Path to the mask data.
            grid_shape_path (str): Path to the shapefile containing grid shapes.
            fenced_area_shape_path (str): Path to the shapefile defining fenced areas.
            batch_size (int): Size of each mini-batch.
            patch_size (int): Size of each patch.
            length (int | None): Length of each training epoch.
            num_workers (int): Number of workers for parallel data loading.
            class_set (int): The high-resolution land cover class set to use (5 or 7).
            use_prior_labels (bool): Whether to use a prior over high-resolution
                                     classes instead of the labels themselves.
            prior_smoothing_constant (float): Smoothing constant added when using prior labels.
            **kwargs (Any): Additional keyword arguments passed to the base class.
        """
        self.raster_image_path = raster_image_path
        self.mask_path = mask_path
        self.grid_shape_path = grid_shape_path
        self.fenced_area_shape_path = fenced_area_shape_path
        self.use_prior_labels = use_prior_labels
        self.class_set = class_set

        super().__init__(
            GarrulusSegmentationDataset,
            batch_size=batch_size,
            patch_size=patch_size,
            length=length,
            num_workers=num_workers,
            **kwargs,
        )

        self.train_aug = AugmentationSequential(
            K.Resize(224),
            K.Normalize(mean=self._mean, std=self._std),
            K.RandomResizedCrop(_to_tuple(self.patch_size), scale=(0.6, 1.0)),
            K.RandomVerticalFlip(p=0.5),
            K.RandomHorizontalFlip(p=0.5),
            data_keys=["image", "mask"],
            extra_args={
                DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None}
            },
        )

    def setup(self, stage: str) -> None:
        """Set up datasets and samplers for training, validation, testing, or prediction.

        Args:
            stage (str): One of 'fit', 'validate', 'test', or 'predict'.
        """
        dataset = GarrulusSegmentationDataset(self.raster_image_path, 
                                              self.mask_path,
                                              **self.kwargs)
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = self.create_dataset_from_tiles(dataset)

        if stage == "fit":
            self.train_batch_sampler = RandomBatchGeoSampler(
                self.train_dataset, self.patch_size, self.batch_size, self.length
            )

        if stage in ["fit", "validate"]:
            self.val_sampler = GridGeoSampler(
                self.val_dataset, self.patch_size, self.patch_size
            )

        # ToDo: split prediction
        if stage in ["test", "predict"]:
            self.test_sampler = GridGeoSampler(
                dataset=self.test_dataset,
                size=self.patch_size,
                stride=self.patch_size,
            )
            self.predict_dataset = self.test_dataset
            self.predict_sampler = self.test_sampler

    def create_dataset_from_tiles(self, dataset):
        """Create a dataset split from grid tiles based on spatial intersections.

        Args:
            dataset: The dataset to be split.

        Returns:
            tuple: A tuple containing the training, validation, and test datasets.
        """
        # load grid shape path
        grid_gdf = gpd.read_file(self.grid_shape_path)
        # Load the field-D fenced area
        fenced_area_gdf = gpd.read_file(self.fenced_area_shape_path)
        fenced_area = fenced_area_gdf.geometry.unary_union
        # Determine the intersecting grid cells with field-D the fenced area
        intersecting_grid_gdf = grid_gdf[grid_gdf.intersects(fenced_area)]

        # Tile indices to for train, test and validation sets
        # test_grid_idx = [50, 51, 52, 53, 54, 55]
        # valid_grid_idx = [57, 58, 59, 60, 61, 62]
        # train_grid_idx = [
        #     3, 9, 10, 11, 16, 17, 18, 19, 23, 24, 25, 26, 27, 29, 30, 31,
        #     32, 33, 34, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48
        # ]

        train_grid_gdf = intersecting_grid_gdf[
            intersecting_grid_gdf["id"].isin(self.train_grid_idx)
        ]
        test_grid_gdf = intersecting_grid_gdf[
            intersecting_grid_gdf["id"].isin(self.test_grid_idx)
        ]
        valid_grid_gdf = intersecting_grid_gdf[
            intersecting_grid_gdf["id"].isin(self.valid_grid_idx)
        ]

        # create roi with BoundixBox list for all the splits
        # todo: repalce BBOX maxt and mint with the raster image maxt and mint
        train_roi_list = [
            BoundingBox(row["left"], row["right"], row["bottom"], row["top"], 0.0, 1e10)
            for _, row in train_grid_gdf.iterrows()
        ]
        test_roi_list = [
            BoundingBox(row["left"], row["right"], row["bottom"], row["top"], 0.0, 1e10)
            for _, row in test_grid_gdf.iterrows()
        ]
        valid_roi_list = [
            BoundingBox(row["left"], row["right"], row["bottom"], row["top"], 0.0, 1e10)
            for _, row in valid_grid_gdf.iterrows()
        ]

        # generate a new dataset by combining the data based on the roi list
        train_data_splits = roi_split(dataset, train_roi_list)
        train_data_union = train_data_splits[0]
        for data_split in train_data_splits[1:]:
            train_data_union = UnionDataset(train_data_union, data_split)

        valid_data_splits = roi_split(dataset, valid_roi_list)
        valid_data_union = valid_data_splits[0]
        for data_split in valid_data_splits[1:]:
            valid_data_union = UnionDataset(valid_data_union, data_split)

        test_data_splits = roi_split(dataset, test_roi_list)
        test_data_union = test_data_splits[0]
        for data_split in test_data_splits[1:]:
            test_data_union = UnionDataset(test_data_union, data_split)

        return train_data_union, valid_data_union, test_data_union

    # def on_after_batch_transfer(
    #     self, batch, dataloader_idx: int
    # ):
    #     """Apply batch augmentations to the batch after it is transferred to the device.
    #     ToDo: maybe be needed later for pre-processing?

    #     Args:
    #         batch: A batch of data that needs to be altered or augmented.
    #         dataloader_idx: The index of the dataloader to which the batch belongs.

    #     Returns:
    #         A batch of data.
    #     """
    #     if self.use_prior_labels:
    #         batch['mask'] = F.normalize(batch['mask'].float(), p=1, dim=1)
    #         batch['mask'] = F.normalize(
    #             batch['mask'] + self.prior_smoothing_constant, p=1, dim=1
    #         ).long()
    #     else:
    #         # replace CUT label with STUMP
    #         if self.class_set == 1:
    #             batch['mask'][batch['mask'] == 1] = 3

    #     return super().on_after_batch_transfer(batch, dataloader_idx)



import os

from lightning.pytorch.cli import ArgsType, LightningCLI

import gdl.datamodules
import gdl.trainers  # noqa: F401
from gdl.datamodules import GarrulusAoiDataModule
from gdl.trainers import GarrulusSemanticSegmentationTask


def main(args: ArgsType = None) -> None:
    """Command-line interface to TorchGeo."""
    # Taken from https://github.com/pangeo-data/cog-best-practices
    rasterio_best_practices = {
        'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
        'AWS_NO_SIGN_REQUEST': 'YES',
        'GDAL_MAX_RAW_BLOCK_CACHE_SIZE': '200000000',
        'GDAL_SWATH_SIZE': '200000000',
        'VSI_CURL_CACHE_SIZE': '200000000',
    }
    os.environ.update(rasterio_best_practices)

    LightningCLI(
        model_class=GarrulusSemanticSegmentationTask,
        datamodule_class=GarrulusAoiDataModule,
        seed_everything_default=0,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={'overwrite': True},
        args=args,
    )

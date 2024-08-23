import abc
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

from torchgeo.datasets import RasterDataset
from torchgeo.datasets import IntersectionDataset

from torchgeo.datasets import (
    BoundingBox,
)
from typing import Any, Callable, Optional


class GarrulusImage(RasterDataset):
    """
    This class is responsible for handling and plotting raster image.
    The original RGB image data consists of 4 bands, and the first 3
    bands are RGB bands.
    """

    # filename_glob = 'd-RGB-9mm*.tif'
    # filename_regex = r'd-RGB-9mm-reference(?!.*mask).*\.tif$'
    filename_regex = r"d-RGB-9mm-reference.tif"
    is_image = True
    separate_files = False
    all_bands = ["B01", "B02", "B03", "B04"]
    rgb_bands = ["B01", "B02", "B03"]

    def plot(self, sample):
        """
        Plot the RGB image from the sample.

        Args:
            sample (dict): A dictionary containing image data with the key 'image'.

        Returns:
            fig (matplotlib.figure.Figure): The figure object containing the plot.
            image (Tensor): The processed image tensor used for plotting.
        """
        rgb_indices = [self.all_bands.index(band) for band in self.rgb_bands]

        # convert to uint8 for plotting
        image = sample["image"][rgb_indices].to(torch.uint8).permute(1, 2, 0)

        # Plot the image
        fig, ax = plt.subplots()
        ax.imshow(image)

        return fig, image


class GarrulusMask(RasterDataset, abc.ABC):
    """
    This class handles the segmentation masks and provides a method
    to plot these masks with appropriate color maps.
    """

    # filename_glob = 'd-RGB-9mm-reference-mask.tif'
    filename_regex = r"d-RGB-9mm-reference-mask.tif"
    is_image = False
    separate_files = False

    cmap = {
        0: (0, 0, 0, 0),
        1: (0, 197, 255, 255),
        2: (50, 50, 78, 255),
        3: (38, 115, 0, 255),
        4: (76, 230, 0, 255),
        5: (163, 255, 115, 255),
    }

    label_names = {
        0: "BACKGROUND",
        1: "CWD",
        2: "MISC",
        3: "CUT",
        4: "STUMP",
        5: "VEGETATION",
    }

    def __init__(self, paths, transforms=None):
        """
        Initialize the GarrulusMask object with file paths and set up the color map.

        Args:
            paths (str): The file path to the mask dataset.
        """
        self.paths = paths
        # defines all cmaps
        colors = [
            (self.cmap[i][0] / 255.0, self.cmap[i][1] / 255.0, self.cmap[i][2] / 255.0)
            for i in range(len(self.cmap))
        ]
        self._cmap = ListedColormap(colors)
        super().__init__(paths, transforms=transforms)

    def plot(self, sample):
        """
        Plot the segmentation mask from the sample.

        Args:
            sample (dict): A dictionary containing mask data with the key 'mask'.

        Returns:
            fig (matplotlib.figure.Figure): The figure object containing the plot.
        """
        mask = sample["mask"].squeeze(0)
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

        axs.imshow(
            mask, vmin=0, vmax=self._cmap.N - 1, cmap=self._cmap, interpolation="none"
        )

        # Show legend
        legend_elements = [
            Patch(
                facecolor=[
                    self.cmap[i][0] / 255.0,
                    self.cmap[i][1] / 255.0,
                    self.cmap[i][2] / 255.0,
                ],
                edgecolor="none",
                label=self.label_names[i],
            )
            for i in self.cmap
        ]
        axs.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=2,
        )
        axs.axis("off")

        return fig


class GarrulusSegmentationDataset(IntersectionDataset):
    """
    This class combines raster images and segmentation masks, providing methods
    for verifying and plotting the dataset.
    """

    label_names = {
        0: "BACKGROUND",
        1: "CWD",
        2: "MISC",
        3: "STUMP",
        4: "VEGETATION",
        # 5: 'CUT',
    }

    cmap = {
        0: (0, 0, 0, 0),
        1: (0, 197, 255, 255),
        2: (100, 100, 78, 255),
        3: (200, 115, 0, 255),
        4: (76, 230, 0, 255),
        # 5: (163, 255, 115, 255),
    }

    def __init__(
        self,
        raster_image_paths="./field-D",
        mask_paths="datasets/garrulus-field-D",
        rgb_bands=["B01", "B02", "B03"],
        grid_shape_path=None,
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    ):
        """
        Initialize the GarrulusSegmentationDataset with image and mask paths.

        Args:
            raster_image_paths (str): Path to the raster images.
            mask_paths (str): Path to the mask images.
            rgb_bands (list): List of bands to use for the RGB image.
        """
        colors = [
            (self.cmap[i][0] / 255.0, self.cmap[i][1] / 255.0, self.cmap[i][2] / 255.0)
            for i in range(len(self.cmap))
        ]
        self._cmap = ListedColormap(colors)

        self.image = GarrulusImage(paths=raster_image_paths, bands=rgb_bands)
        self.mask = GarrulusMask(paths=mask_paths)
        self.grid_shape_path = grid_shape_path
        self.transforms = transforms

        super().__init__(self.image, self.mask, transforms=transforms)

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve image and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of data/labels and metadata at that index

        Raises:
            IndexError: if query is not within bounds of the index
        """
        if not query.intersects(self.bounds):
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        # All datasets are guaranteed to have a valid query
        samples = [ds[query] for ds in self.datasets]

        sample = self.collate_fn(samples)

        if self.transforms is not None:
            # create new sample and remove crs and bbox before applying transforms
            new_sample = sample.copy()
            new_sample.pop("crs")
            new_sample.pop("bbox")
            new_sample = self.transforms(new_sample)
            sample["image"] = new_sample["image"].squeeze()
            sample["mask"] = new_sample["mask"].squeeze()

        return sample

    def _verify(self):
        """
        Verify that the dataset is valid by checking the checksums.
        """
        raise NotImplementedError

    def plot(self, sample, show_titles=True, suptitle=None):
        """
        Plot a sample from the dataset.

        Args:
            sample (dict): A sample returned by RasterDataset.__getitem__.
            show_titles (bool): Flag indicating whether to show titles above each panel.
            suptitle (str | None): Optional string to use as a suptitle.

        Returns:
            fig (matplotlib.figure.Figure): A matplotlib Figure with the rendered sample.
        """
        if sample["image"].shape[1] > 3:
            rgb_indices = []
            for band in self.image.rgb_bands:
                if band in self.image.bands:
                    rgb_indices.append(self.image.bands.index(band))
                else:
                    raise ValueError("RGB band does not include all RGB bands")

            image = sample["image"][rgb_indices].permute(1, 2, 0)
        else:
            image = sample["image"].permute(1, 2, 0)

        # Stretch to the full range
        image = (image - image.min()) / (image.max() - image.min())

        mask = sample["mask"].numpy().astype("uint8").squeeze()

        num_panels = 2
        showing_predictions = "prediction" in sample
        if showing_predictions:
            predictions = sample["prediction"].numpy().astype("uint8").squeeze()
            num_panels += 1

        fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 4, 5))
        axs[0].imshow(image)
        axs[0].axis("off")

        axs[1].imshow(mask, vmin=0, vmax=self._cmap.N - 1, cmap=self._cmap)
        axs[1].axis("off")

        # Show legend
        legend_elements = [
            Patch(
                facecolor=[
                    self.cmap[i][0] / 255.0,
                    self.cmap[i][1] / 255.0,
                    self.cmap[i][2] / 255.0,
                ],
                edgecolor="none",
                label=self.label_names[i],
            )
            for i in self.cmap
        ]
        axs[1].legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=2,
        )

        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if showing_predictions:
            axs[2].imshow(predictions, vmin=0, vmax=self._cmap.N - 1, cmap=self._cmap)
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig

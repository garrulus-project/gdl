import numpy as np
import rasterio
import os
import geopandas as gpd
from rasterio.features import geometry_mask


def create_mask(raster_image_source, geopackages, mask_output_path="mask_output"):
    """
    Create a raster image mask given geopackage labels.

    Args:
        raster_image_source (str): Path to the raster image file.
        geopackages (dict): A dictionary containing geopackage information, where the key is the class name,
                            and the value is a dictionary with the label mask and path to the geopackage file.
        mask_output_path (str): Directory where the output mask will be saved. Defaults to "mask_output".

    Raises:
        ValueError: If the geopackage path does not exist.
    """
    os.makedirs(mask_output_path, exist_ok=True)

    # Open the raster file
    with rasterio.open(raster_image_source) as src:
        # Initialize the mask to be background (all zeros)
        raster_image_mask = np.zeros((src.height, src.width), dtype=np.uint8)

        for class_name, class_info in geopackages.items():
            mask_label = class_info["label"]
            gpkg_path = class_info["path"]

            if not os.path.exists(gpkg_path):
                raise ValueError(f"Path does not exist: {gpkg_path}")

            # Read the GeoPackage file
            geo = gpd.read_file(gpkg_path)

            # Create a mask from the geometries in the geopackage
            mask = geometry_mask(
                geo.geometry,
                out_shape=(src.height, src.width),
                transform=src.transform,
                invert=True,
            )

            raster_image_mask[mask] = mask_label

            print(f"{class_name} RGB color: {mask_label}")

        # Save the combined mask as a GeoTIFF file
        output_filename = os.path.splitext(os.path.basename(raster_image_source))[0]
        output_path = os.path.join(mask_output_path, f"{output_filename}_mask.tif")

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=raster_image_mask.shape[0],
            width=raster_image_mask.shape[1],
            nodata=0,
            count=1,
            dtype=np.uint8,
            crs=src.crs,
            transform=src.transform,
            compress="none",
        ) as dst:
            # Write the mask to the GeoTIFF file
            dst.write(raster_image_mask, 1)

        print(f"Raster image mask is saved to: {output_path}")


def create_intersecting_grids(
    grid_path,
    fenced_area_path,
):
    """
    Create intersecting grid cells within a fenced area.
    Args:
        grid_path (str): Path to the grid file.
        fenced_area_path (str): Path to the fenced area file.

    Returns:

    """
    # Load the grid shapefile
    grid_gdf = gpd.read_file(grid_path)

    # Load the fenced area
    fenced_area_gdf = gpd.read_file(fenced_area_path)

    # Assuming the land area shapefile contains one polygon/multipolygon
    fenced_area = fenced_area_gdf.geometry.union_all()

    # Determine the intersecting grid cells
    intersecting_grids = grid_gdf[grid_gdf.intersects(fenced_area)]

    return intersecting_grids


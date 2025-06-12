"""Main module for my response to the KoBold challenge."""
from contextlib import contextmanager
from io import BytesIO
import logging
from time import perf_counter
from typing import Optional, Tuple

from affine import Affine
import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from rapidfuzz.fuzz import partial_ratio
import rasterio
from rasterio.features import rasterize
import rasterio.transform
import shapely

log = logging.getLogger(__name__)


def _save_matplotlib_to_bytes(fig: plt.Figure) -> bytes:
    """Save a Matplotlib figure to bytes as a png."""
    buffer = BytesIO()
    fig.savefig(buffer, bbox_inches='tight', pad_inches=0)
    buffer.seek(0)
    return buffer.read()


def classify_rock_type(
        gdf: gpd.GeoDataFrame,
        type: str,
        fuzzy_match_pct: Optional[float] = None,
        new_col_name: Optional[str] = None) -> gpd.GeoDataFrame:
    """Append a boolean column to a GeoDataFrame of bedrock polygons indicating a particular type of rock.

    Uses fuzzy matching to capture mistyped or differently described versions of the rock.

    Parameters
    ----------
    gdf: gpd.GeoDataFrame
        The input dataframe, loaded from a bedrock polygon file.
    type: str
        The type of rock to classify. ALl of the fields within the bedrock polygon file are searched for this string.
    fuzzy_match_pct: Optional[float]
        A floating point number in the range 0-100 or None. If specified, fuzzy matching will be used.
    new_col_name: Optional[str]
        The name of the column added to the dataframe before it is returned,
        indicating that a row matches the type parameter in one or more columns.
    Returns
    -------
    gpd.GeoDataFrame
        an augmented version of the input GeoDataFrame with an additional column indicating a classified rock.
    """
    if fuzzy_match_pct is not None:
        assert 0 < fuzzy_match_pct <= 100, f"fuzzy_match_pct must be between 0 and 100 but was {fuzzy_match_pct}"
    else:
        fuzzy_match_pct = 100

    if new_col_name is None:
        new_col_name = f'is_{type}'

    gdf[new_col_name] = gdf.map(
        lambda cell: isinstance(cell, str) and partial_ratio(
            type, cell, processor=lambda x: x.lower()) >= fuzzy_match_pct).any(axis=1)

    log.debug(f'matched {gdf[new_col_name].sum()} of {len(gdf)} rows at a fuzzy match percentage of {fuzzy_match_pct}')

    return gdf


def visualise_adjacent_rocks(
        gdf: gpd.GeoDataFrame,
        is_rock_a_col_name: str,
        is_rock_b_col_name: str,
        max_separation_m: float,
        plot_alpha: float = 0.5,
        plot_title: str = "Rock adjacency visualisation") -> bytes:
    """Visualise adjacent rock types within a geodataframe of bedrock polygons.

    The result is saved as a png, and returned as bytes.

    Parameters
    ----------
    gdf: gpd.GeoDataFrame
        The GeoDataFrame containing the bedrock polygons to visualise
    is_rock_a_col_name: str
        The column name within gdf for a boolean column indicating that rows are parts of rock type A.
    is_rock_b_col_name: str
        The column name within gdf for a boolean column indicating that rows are parts of rock type B.
    max_separation_m: float
        A polygon showing areas where rock types A and B are separated by up to this many meters
        is added to the visualisation.
    plot_alpha: float
        A value between 0 and 1 determining the transparency of the polygons on the plots
    plot_title: str
        The title of the plot.
    
    Returns
    -------
    bytes
        The png image generated as bytes, which can then be saved to file.
    """
    assert is_rock_a_col_name in gdf.columns, "is_rock_a_col_name not in gdf.columns"
    assert is_rock_b_col_name in gdf.columns, "is_rock_b_col_name not in gdf.columns"

    assert gdf.crs is not None, "GeoDataFrame has no CRS defined"
    assert all(axis.unit_name == "metre" for axis in gdf.crs.axis_info), "CRS is not in meters"

    assert gdf.crs.to_epsg() is not None, "GeoDataFrame's CRS cannot be converted into an EPSG code"

    is_rock_a_geom = gdf.loc[gdf[is_rock_a_col_name], 'geometry'].union_all()
    is_rock_b_geom = gdf.loc[gdf[is_rock_b_col_name], 'geometry'].union_all()

    max_separation_poly = shapely.intersection(
        is_rock_a_geom.buffer(max_separation_m / 2),
        is_rock_b_geom.buffer(max_separation_m / 2)
    )

    fig, ax = plt.subplots(figsize=(20,20))

    gpd.GeoSeries(is_rock_a_geom, crs=gdf.crs).plot(ax=ax, color='blue', alpha=plot_alpha)
    gpd.GeoSeries(is_rock_b_geom, crs=gdf.crs).plot(ax=ax, color='red', alpha=plot_alpha)
    gpd.GeoSeries(max_separation_poly, crs=gdf.crs).plot(ax=ax, color='green', alpha=plot_alpha)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=gdf.crs)

    ax.set_title(plot_title, fontsize=14)
    fig.tight_layout()
    ax.grid()
    ax.set_xlabel(f' EPSG:{gdf.crs.to_epsg()} X (m)')
    ax.set_ylabel(f' EPSG:{gdf.crs.to_epsg()} Y (m)')

    return _save_matplotlib_to_bytes(fig)


def calculate_fine_separation(
        gdf: gpd.GeoDataFrame,
        is_rock_a_col_name: str,
        is_rock_b_col_name: str,
        max_separation_m: float,
        raster_resolution_m: float,
        separation_resolution_m: float) -> Tuple[np.ndarray, Affine]:
    """Calculate the separation between two types of rock at all points within the bounds of the provided GeoDataFrame.

    The result is returned as a raster dataset.

    Parameters
    ----------
    gdf: gpd.GeoDataFrame
        The GeoDataFrame containing the bedrock polygons for which to calculate the separation
    is_rock_a_col_name: str
        The column name within gdf for a boolean column indicating that rows are parts of rock type A.
    is_rock_b_col_name: str
        The column name within gdf for a boolean column indicating that rows are parts of rock type B.
    max_separation_m: float
        A polygon showing areas where rock types A and B are separated by up to this many meters
        is added to the visualisation.
    raster_resolution_m: float
        The resolution in meters for which to calculate the raster. Appropriate values are
        typically one thousandth to one ten thousandth the width of the input gdf.
    separation_resolution_m: float
        The resolution in meters at which the separation is quantised for calculation speed.
        Appropriate values are typically one tenth to one hundredth max_resolution_m

    Returns
    -------
    np.ndarray
        A grid which records the separation between the two kinds
        of rock at every point where that separation is less than or equal to max_separation_m.
    Affine
        The affine transform mapping the raster into the crs of the provided gdf.
    """
    assert is_rock_a_col_name in gdf.columns, "is_rock_a_col_name not in gdf.columns"
    assert is_rock_b_col_name in gdf.columns, "is_rock_b_col_name not in gdf.columns"

    assert gdf.crs is not None, "GeoDataFrame has no CRS defined"
    assert all(axis.unit_name == "metre" for axis in gdf.crs.axis_info), "CRS is not in meters"

    assert gdf.crs.to_epsg() is not None, "GeoDataFrame's CRS cannot be converted into an EPSG code"

    assert raster_resolution_m > 0, "raster_resolution_m must be greater than zero"
    assert separation_resolution_m > 0, "separation_resolution_m must be greater than zero"

    is_rock_a_geom = gdf.loc[gdf[is_rock_a_col_name], 'geometry'].union_all()
    is_rock_b_geom = gdf.loc[gdf[is_rock_b_col_name], 'geometry'].union_all()

    max_separation_poly = shapely.intersection(
        is_rock_a_geom.buffer(max_separation_m / 2),
        is_rock_b_geom.buffer(max_separation_m / 2)
    )

    # use this polygon to establish bounds for the separation map
    sepmap_minx, sepmap_miny, sepmap_maxx, sepmap_maxy = max_separation_poly.bounds

    # add a small buffer for ease of use
    sepmap_minx -= 2 * max_separation_m
    sepmap_miny -= 2 * max_separation_m
    sepmap_maxx += 2 * max_separation_m
    sepmap_maxy += 2 * max_separation_m

    # Calculate raster dimensions and projection
    sepmap_width_pixels = int(np.ceil((sepmap_maxx - sepmap_minx) / raster_resolution_m))
    sepmap_height_pixels = int(np.ceil((sepmap_maxy - sepmap_miny) / raster_resolution_m))
    log.debug(f'separation raster is {sepmap_width_pixels}x{sepmap_height_pixels} pixels')
    sepmap_transform = Affine(raster_resolution_m, 0, sepmap_minx, 0, -raster_resolution_m, sepmap_maxy)

    # Create empty raster array
    sepmap = np.full((sepmap_height_pixels, sepmap_width_pixels), fill_value=np.nan, dtype=np.float32)

    separation_levels_m = np.arange(0, max_separation_m, separation_resolution_m)

    # starting from the largest separation value (lowest probability) and working down,
    # calculate the appropriate polygon using buffering, then assign all points within
    # that buffer a lower separation value / higher probability
    for d in sorted(separation_levels_m, reverse=True):
        is_rock_types_within_distance_poly = shapely.intersection(
            is_rock_a_geom.buffer(d / 2),
            is_rock_b_geom.buffer(d / 2)
        )

        is_rock_types_within_distance_raster = rasterize(
            [(is_rock_types_within_distance_poly, 1)],
            out_shape=sepmap.shape,
            transform=sepmap_transform,
            dtype="uint8"
        )

        # update pixel values covered by this polygon
        sepmap[is_rock_types_within_distance_raster == 1] = d

    log.debug(f'median separation after calculation is {np.median(sepmap.ravel()):.2f} m')

    return sepmap, sepmap_transform


def visualise_geotiff(
        raster: rasterio.DatasetReader,
        transform: Affine,
        crs_epsg: str,
        plot_title: str,
        colorbar_label: str,
        plot_alpha: float = 0.5,
        colormap: str = "plasma_r",
        contextily_zoom: Optional[int] = None) -> bytes:
    """Generate a png image visualising a GeoTIFF.

    This plot is intended for human consumption / use in presentations, not for machine use.
    Lower probability values are faded through to transparent so as not to ide the context map.
    
    Parameters
    ----------
    raster: np.ndarray
        The raster data to save to disk
    transform: Affine
        The transform mapping the data into its target crs.
    crs_epsg: str
        The coordinate reference system for the data as an EPSG code.
    plot_title: str
        The title of the plot.
    colorbar_label: str
        The lable of the colorbar included with the plot.
    plot_alpha: float
        A value between 0 and 1 determining the transparency of the heatmap on the plot
    colormap: str
        The colormap used on the plot.
    contextily_zoom: Optioanl[int]
        The zoom level to use for the plot background.

    Returns
    -------
    bytes
        A rendered png image
    """
    fig, ax = plt.subplots(figsize=(20, 20))

    minx, miny, maxx, maxy = rasterio.transform.array_bounds(raster.shape[0], raster.shape[1], transform)

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    # add a basemap, setting the zoom manually so the result is a bit nicer
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=crs_epsg, zoom=contextily_zoom)

    cmap = plt.get_cmap(colormap)

    # set the alpha channel to the value of the heatmap, so that low probabilities fade to the background image
    heatmap_for_plotting = cmap(raster)
    heatmap_for_plotting[..., -1] = raster

    # overplot heatmap
    img = ax.imshow(
        heatmap_for_plotting,
        extent=(minx, maxx, miny, maxy),  # note different order to bounds!
        origin="upper",
        cmap=cmap,
        alpha=plot_alpha
    )

    ax.set_title(plot_title, fontsize=14)
    fig.tight_layout()
    ax.grid()
    ax.set_xlabel(f' EPSG:{crs_epsg} X (m)')
    ax.set_ylabel(f' EPSG:{crs_epsg} Y (m)')
    fig.colorbar(img, ax=ax, label=colorbar_label)

    return _save_matplotlib_to_bytes(fig)


def simple_sigmoid(dist: np.ndarray, max_dist: float, steepness: float = np.nan) -> np.ndarray:
    """Simple sigmoid function.
    
    Used to convert rock separation distance into likelihood.
    
    Parameters
    ----------
    dist: np.ndarray
        numpy array representing a raster of calculated distances between two different rock types.
    max_dict: float
        The maximum distance. Beyond this, the function will return zero.
    steepness: float
        The k parameter, allowing the tuning of the sigmoid. If one is not provided,
        a "sensible" k will be inferred from max_dist.

    Returns
    -------
    np.ndarray
        A numpy array with the same shape as dist containing values in between 0 and 1.
    """
    # work out a sensible steepness if not provided:
    if np.isnan(steepness):
        steepness = 2 / max_dist * np.log((1 / 1e-3)-1)
    output = np.zeros(dist.shape)
    output[dist <= max_dist] = 1 / (1 + np.exp(steepness*(dist[dist <= max_dist].ravel()-max_dist/2)))
    return output.reshape(dist.shape)


@contextmanager
def timer(label: str = ""):
    """A simple timer implemented as a context manager"""
    start = perf_counter()
    yield
    end = perf_counter()
    log.info(f"{label} Elapsed: {(end - start)*1e3:.2f} ms")


def save_raster_to_disk(raster: np.ndarray, transform: Affine, crs_epsg: str, filename: str) -> None:
    """Save a raster to disk as a GeoTIFF.
    
    Parameters
    ----------
    raster: np.ndarray
        The raster data to save to disk
    transform: Affine
        The transform mapping the data into its target crs.
    crs_epsg: str
        The coordinate reference system for the data as an EPSG code.

    """
    with rasterio.open(
        filename,
        'w',
        driver='GTiff',
        height=raster.shape[0],
        width=raster.shape[1],
        count=1,
        dtype='float32',
        crs=crs_epsg,
        transform=transform
    ) as dataset:
        dataset.write(raster, 1)

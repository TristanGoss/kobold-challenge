import logging

import geopandas as gpd
import rasterio

from kobold_challenge import tools

logging.basicConfig(
    level='INFO',
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)

log = logging.getLogger(__name__)

def main():
    """Main script for the Kobold challenge.
    
    This script illustrates how to use the provided tools.
    """

    # To begin with, load your bedrock geopackage from file, as specified in the problem introduction.
    with tools.timer("loading the bedrock geopackage"):
        bedrock_data = gpd.read_file('BedrockP.gpkg')
    
    # We want to identify areas that may contain Cobalt.
    # Cobalt is found in (amongst many other places) regions where serpentinite
    # and other ultramafic rocks are close to granodiorite rocks.
    # Try passing fuzzy_match_pct=80 to the below function to uncover some other deposits, perhaps?
    # That would need discussion with a geologist.
    # To identify these regions, we must first correctly classify the bedrock polygons:
    with tools.timer("classifying the rock types"):
        for rock_type in ['serpentinite', 'granodiorite', 'ultramafic']:
            bedrock_data = tools.classify_rock_type(
                gdf=bedrock_data, type=rock_type, new_col_name=f'is_{rock_type}')
    
    # Let's amalgamate the serpentinite and ultramafic results together, as advised in the instructions
    bedrock_data['is_serpentinite_or_ultramafic'] = \
        bedrock_data['is_serpentinite'] & bedrock_data['is_ultramafic']
    
    # Let's generate an image so we can check that the output looks as we would expect,
    # given our expert knowledge of the region. We're only interested in areas
    # where these rocks are within 10km of each other.
    max_separation_m = 10e3

    with tools.timer("visualising the rock type adjacency"):
        with open('rock_adjacency_visualisation.png', 'wb') as fh:
            fh.write(
                tools.visualise_adjacent_rocks(
                    gdf=bedrock_data,
                    is_rock_a_col_name='is_serpentinite_or_ultramafic',
                    is_rock_b_col_name='is_granodiorite',
                    max_separation_m=max_separation_m))
        
    # Next, let's calculate the separation between these rock types at all relevant points.
    # We use a 100m resolution, whcih results in a roughly 4k raster, and 200m separation resolution,
    # which is adequate to map out the probability curve.
    with tools.timer("calculating rock type separation at 100m spatial, 200m separation resolution"):
        sepmap, transform = tools.calculate_fine_separation(
            gdf=bedrock_data,
            is_rock_a_col_name='is_serpentinite_or_ultramafic',
            is_rock_b_col_name='is_granodiorite',
            max_separation_m=max_separation_m,
            raster_resolution_m=100,
            separation_resolution_m=200,
        )

    # Let's save that to file so we can use it later / in other software.
    epsg = bedrock_data.crs.to_epsg()
    tools.save_raster_to_disk(
        sepmap, transform, epsg, "cobalt_rock_separation.tif")

    # Next, we want to convert this separation to probability.
    # Let's use a simple sigmoid for this.
    cobalt_map = tools.simple_sigmoid(sepmap, max_separation_m)

    # Let's save this result as well.
    tools.save_raster_to_disk(
        cobalt_map, transform, epsg, "cobalt_likelihood.tif")

    # Let's visualise the result
    with tools.timer("visualising finished heatmap"):
        with open("cobalt_heatmap_visualisation.png", "wb") as fh:
            fh.write(tools.visualise_geotiff(
                cobalt_map,
                transform,
                epsg,
                plot_title="Likelihood of Cobalt deposit i.v.o Squamish, Canada",
                colorbar_label="Likelihood of Cobalt deposit",
                contextily_zoom=9))
    
    # Lastly, how do we look up a value for a specific point?
    # This is easy, becuase we have the data as a GeoTIFF.
    # We simply open the GeoTIFF and perform a nearest neighbour lookup.
    # This is adequate because our raster resolution is already 100m,
    # and other issues may restrict core drilling at resolutions lower than this.
    query_x, query_y = 540e3, 5.6e6
    with rasterio.open("cobalt_likelihood.tif") as src:
        value = next(src.sample([(query_x, query_y)]))[0]

    log.info(f'likelihood of finding Cobalt at x={query_x}, y={query_y} in EPSG {epsg} is {value:.2f}')


if __name__ == "__main__":
    main()

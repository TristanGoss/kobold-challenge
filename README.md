# kobold-challenge

This is a response to an interview "take-home problem" posed by [KoBold metals](https://www.koboldmetals.com/).

## Challenge Description
The objective of this exercise is to create a dataset in the form of a heat map representing the likelihood of finding a cobalt deposit at each point on the map. The heat maps should be stored in such a way that we could easily view the numeric value for the likelihood of finding cobalt at each and every location.

### Tasks
To determine where cobalt is likely to occur, we are going to assume that cobalt deposits are found in the presence of two different types of rock. We are interested in locations where these rock types are in contact with each other, or in close proximity to one another. The likelihood of cobalt occurring should fall off smoothly to zero where distance between the two rock types exceeds some distance (~10 km).
1. Build a software tool capable of creating heat maps based on the proximity of any two rock types, with an adjustable fall-off distance parameter.
2. Generate a heat map based on proximity of 
**serpentinite** and **granodiorite**. Serpentinite is a subtype of “ultramafic” rock; so here, for the purpose of this exercise, assume that all ultramafic rocks are also serpentinite.
### Data
We have provided a set of data that shows the type of bedrock (what lies beneath the soil) at each location in an example region. This data comes from the British Columbia Geological Survey. The coordinates in this dataset are provided in the 
coordinate reference system EPSG:26910 which has units meters. Allowing distances between points to be calculated in meters without any conversions.

The map is provided here as a shapefile (BedrockP.shp). BedrockP.shp consists of a set of polygons that collectively cover the area and a table of attributes associated with each polygon.

## Tristan's Response

### Installation
My response to this challenge is provided in the form of a python package designed to be installed using the python package manager [Poetry](https://python-poetry.org/).

To install my solution, first install Poetry, then clone this repository and run `poetry install --only main`.
If instead you want to develop it (or run the Jupyter notebooks), then instead run `poetry install --with dev`.

### Use
To run the solution, place the `BedrockP.gpkg` file in the working directory and run `poetry run challenge`. This simple script shows how to use the provided tools.

In order to return the solution quickly, I have not provided a full UI.

### Quality
Ruff linting is in place (autoformat with `poetry run ruff format .` and link with `poetry run ruff check --fix .`), but as per the instructions, no other quality steps have been completed, nor have any unit tests be provided. Things that would need to happen to make this code production-ready:
- Unit tests
- Much more interaction with the end user to understand what they want
- Adopting the linting, documentation and naming standards of the rest of the codebase
- Improved error management, in line with the rest of the codebase
- Integration into existing system and integration tests.
- Peer review
- QA approval

### Algorithm description
The algorithm in use is as follows:
1. Use fuzzy text matching across all columns within the provided geopackage to classify individual polygons based on user requirements. Polygons classified as each rock type are then unioned to produce one large multipolygon for each rock type of interest. Basic polygon set operations are used to handle the "serpentinite" vs “ultramafic” issue.
2. Sticking with polygons, we buffer (i.e. dilate) the boundaries of the two rock types by half the maximum separation, then intersect the two. This provides a multipolygon covering all points of relevance to the remainder of the algorithm.
3. To compute the heatmap, we first rasterise the intersection of the buffered rock type boundaries and assign all pixels outside of that intersection multipolygon a value of NaN. All pixels within it are assigned a value of the maximum separation distance.
4. We then reduce the buffer distance, recalculate the intersection, and repeat the rasterisation of the intersection polygon a number of times, assigning all pixels covered by the new, smaller intersection polygon a progressively reduced separation distance each time. Working with pixels in this way ensures we will never have floating point errors associated with adjacent polygons, so all points will have a valid value. At the end of this process, all pixels have been assigned a valid separation distance (between the two rock types). The calculation is not exact, but it avoids the expensive calculation of distances between pixels and polygons, instead using fast polygon rasterisation algorithms only.
5. We then bulk convert the separation distances into Cobalt deposit likelihoods, and generate two figures; a png for human consumption and a GeoTIFF for machine use. As the GeoTIFF is ~4k resolution, there is likely enough accuracy for a simple nearest neighbour lookup to be adequate for later use.

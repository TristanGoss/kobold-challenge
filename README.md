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

## Use
To run the solution, place the `BedrockP.gpkg` file in the working directory and run `poetry run challenge`. This simple script shows how to use the provided tools.

In order to return the solution quickly, I have not provided a full UI.

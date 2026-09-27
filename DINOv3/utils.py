import sys

import numpy as np
import geopandas as gpd

from shapely.geometry import Polygon

from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler

from shapely.geometry import LineString
import matplotlib
import matplotlib.pyplot as plt


def calculate_utm_epsg(longitude, latitude):
    utm_zone = int((longitude + 180) / 6) + 1
    if latitude >= 0:
        epsg_code = 32600 + utm_zone
    else:
        epsg_code = 32700 + utm_zone
    return epsg_code


def lon_lat_to_pixel(coord, minlon, maxlon, minlat, maxlat,
                     col, col_min, row, row_min, mask_width, mask_height):
    x = (coord[0] - minlon) * (256 / (maxlon - minlon))
    y = (coord[1] - minlat) * (256 / (maxlat - minlat))
    temp_x = x + (col - col_min) * 256
    temp_y = y + (row - row_min) * 256
    temp_x = max(0, min(temp_x, 256 * mask_width))
    temp_y = max(0, min(temp_y, 256 * mask_height))
    return temp_x, temp_y


def deg2XY(lon_deg, lat_deg, zoom=15):
    lat_rad = np.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - np.log(np.tan(lat_rad) + (1 / np.cos(lat_rad))) / np.pi) / 2.0 * n)
    return xtile, ytile


def XY2deg_batch(x_arr, y_arr, zoom=15):
    n = 2.0 ** zoom
    lon_deg_arr = x_arr / n * 360.0 - 180.0
    lat_rad_arr = np.arctan(np.sinh(np.pi * (1 - 2 * y_arr / n)))
    lat_deg_arr = np.rad2deg(lat_rad_arr)
    return lon_deg_arr, lat_deg_arr


def compute_tile_coordinates(min_x, max_x, min_y, max_y):
    x_arr = np.arange(min_x, max_x + 1)
    y_arr = np.arange(min_y, max_y + 1)
    lon_arr, lat_arr = XY2deg_batch(x_arr, y_arr)
    return lon_arr, lat_arr, x_arr, y_arr


def create_tile_polygons(lon_arr, lat_arr, x_arr, y_arr):
    lon_mesh, lat_mesh = np.meshgrid(lon_arr, lat_arr, indexing='ij')
    x_mesh, y_mesh = np.meshgrid(x_arr, y_arr, indexing='ij')

    vertices = np.array([
        lon_mesh[:-1, :-1], lat_mesh[:-1, :-1],
        lon_mesh[1:,  :-1], lat_mesh[1:,  :-1],
        lon_mesh[1:,  1:],  lat_mesh[1:,  1:],
        lon_mesh[:-1, 1:],  lat_mesh[:-1, 1:]
    ])
    vertices = vertices.reshape(4, 2, -1)
    vertices = np.transpose(vertices, axes=(2, 0, 1))
    polygons = [Polygon(p) for p in vertices]

    vertices_x_y = np.array([
        x_mesh[:-1, :-1], y_mesh[:-1, :-1],
        x_mesh[1:,  :-1], y_mesh[1:,  :-1],
        x_mesh[1:,  1:],  y_mesh[1:,  1:],
        x_mesh[:-1, 1:],  y_mesh[:-1, 1:]
    ])
    vertices_x_y = vertices_x_y.reshape(4, 2, -1)
    vertices_x_y = np.transpose(vertices_x_y, axes=(2, 0, 1))
    y_x = [f"{int(p[0][1])}_{int(p[0][0])}" for p in vertices_x_y]

    tile_gpd = gpd.GeoDataFrame({"Y_X": y_x}, geometry=polygons, crs="EPSG:4326")
    return tile_gpd


def geometry_to_listXY(geometry):
    min_x, min_y, max_x, max_y = geometry.bounds
    minx, maxy = deg2XY(min_x, min_y)
    maxx, miny = deg2XY(max_x, max_y)

    lon_arr, lat_arr, x_arr, y_arr = compute_tile_coordinates(minx, maxx, miny, maxy)
    tile_gpd = create_tile_polygons(lon_arr, lat_arr, x_arr, y_arr)

    intersection = gpd.sjoin(
        tile_gpd,
        gpd.GeoDataFrame(data={"tmp": [0]}, geometry=[geometry], crs="EPSG:4326").drop(columns=["tmp"]),
        predicate='intersects',
        how='inner'
    )
    Y_X = list(intersection.Y_X)

    if len(Y_X) < 2:
        for x in range(minx, maxx + 1):
            for y in range(miny, maxy + 1):
                Y_X.append(f"{y}_{x}")

    return Y_X


def get_YX_area(area_shp):
    area_shp = area_shp.to_crs(epsg=4326)
    geometries = area_shp["geometry"]
    area_shp["Y_X"] = geometries.map(geometry_to_listXY)
    Y_X = sum(list(area_shp["Y_X"]), [])
    Y_X = sorted(list(set(Y_X)))
    return area_shp, Y_X


def XY2deg(x, y, zoom=15):
    n = 2.0 ** zoom
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * y / n)))
    lat_deg = np.rad2deg(lat_rad)
    return lat_deg, lon_deg

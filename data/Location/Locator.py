from re import search
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean as distance

LOCATION_TABLE = pd.read_csv("data/Location/toponym_cordinate.csv",index_col="Name")

# def _convert_coordinate(coordinate: str) -> np.array:
#     '''HELPER: get a Coordinate as a string and returns it as numpy array

#     :param coordinate: Coordinate to convert
#     :type coordinate: str
#     :return: Coordinate as numpy array if it is a list of points, None otherwise
#     :rtype: np.array
#     '''
#     Coordinate = eval(coordinate)
#     return np.array(Coordinate) if isinstance(Coordinate, list) and len(Coordinate) == 2 and search(
#         r'\d+\.\d+|\b\d+\b', Coordinate[0]) and search(r'\d+\.\d+|\b\d+\b', Coordinate[1]) else None # checks if it is list of numbers as str
#     # TODO: כמה זה נורא ^

def _get_distance(location1: pd.Series, location2: pd.Series):
    '''HELPER: this function gets as input 2 rows of the LOCATION_TABLE and returns the eucalidean distance between them

    :param location1: one row from the table mention above
    :type location1: pd.Series
    :param location2: another row from the table mention above
    :type location2: pd.Series
    :return: eucalidean distance between the two coordinates
    :rtype: float
    '''    
    return distance(location1.loc['x':'y'], location2.loc['x':'y'])

# def _convert_coordinate ():


if __name__ == "__main__":
    print("yes")

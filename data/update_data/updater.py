from data.Finished.PreWork.data_tabler import *
import glob
import urllib3
import pandas as pd
import os
import sys
import bs4
import json
import zipfile
import io
import threading
sys.path.append(os.path.abspath(os.path.curdir))

FILES_FOR_PROJECT = {}


def get_project_list() -> list:
    '''get_project_list returns list of the folders in the 'jsons_unzipped' folder. 
    the assumption is 'jsons_unzipped' contains only folder. if the folder is empty, it returns an empty list.

    :return: a list of folders in 'jsons_unzipped' folder. if it empty, it returns an empty list
    :rtype: list
    '''
    return os.path.listdir("jsons_unzipped") if os.path.isdir("jsons_unzipped") else []


def download_project_list(subproject: bool = False) -> None:
    '''download_project_list downloads the project zip of jsons from the oracc web corpus

    :param subproject: switch for cataloging by subprojects or not, defaults to False
    :type subproject: bool, optional
    '''
    web = bs4.BeautifulSoup(urllib3.PoolManager().request(
        "GET", "http://oracc.museum.upenn.edu/projectlist.html").data.decode("utf-8"))
    for link in web.find_all("div", attrs={"class": "project-entry"}):
        try:
            data = urllib3.PoolManager().request(
                "GET", f'{link.find("a").attrs["href"].replace(".", "http://oracc.museum.upenn.edu/")}/json').data
            Data = io.BytesIO(data)
            threading.Thread(None, zipfile.ZipFile(Data, "r").extractall, args=(("jsons_unzipped"),)).start()
        except urllib3.exceptions.MaxRetryError:
            continue
        except zipfile.BadZipfile:
            continue
    if subproject:
        for link in web.find_all("div", attrs={"class": "subproject-entry"}):
            try:
                data = urllib3.PoolManager().request(
                    "GET", f'{link.find("a").attrs["href"].replace(".", "http://oracc.museum.upenn.edu/")}/json').data
                Data = io.BytesIO(data)
                threading.Thread(None, zipfile.ZipFile(Data, "r").extractall, args=(("jsons_unzipped"),)).start()
            except urllib3.exceptions.MaxRetryError:
                continue
            except zipfile.BadZipfile:
                continue
    # TODO: להסביר שהייתי חייב להתייחס בנאמנות מלאה לקובץ הקטלוג, כי אין לי מושג איך לשלוף את המידע ישירות מהאתר דרך הjs שלו


def _get_catalogs() -> list:
    '''_get_catalogs finds the catalog files and returns their paths

    :return: list of paths of the catalog files
    :rtype: list
    '''
    paths = []
    for List in (glob.glob(f"{'*/'*i}catal*.json", recursive=True) for i in range(1, 6)
                 if glob.glob(f"{'*/'*i}catal*.json", recursive=True)):
        paths += List
    FILES_FOR_PROJECT = {p.split('\\')[1] for p in paths}
    return paths


def read_catalogs(path: str = None) -> None:
    '''read_catalogs reads the catalog from the folders, and gets the members. than it saves them to the global variable
    FILES_FOR_PROJECT

    :param path: path of the catalog, defaults to None
    :type path: str, optional
    '''    
    if not path:
        paths = _get_catalogs()
    else:
        paths = [path, ]
    for p in paths:
        with open(p, "r", encoding="utf_8") as files:
            try:
                FILES_FOR_PROJECT[p.split("\\")[1]] = json.loads(files)["members"]
            except KeyError:
                continue


def download_xml() -> None:
    '''download_xml downloads the xml file from the oracc website, as they retrived from the catalog
    '''
    for f in FILES_FOR_PROJECT:
        if not os.path.exists(f'{f}/xml'):
            os.makedirs(f'{f}/xml')
        for member in FILES_FOR_PROJECT[f]:
            try:
                with open(f"{f}/xml/{member}.xml", "wb") as xml_file:
                    Data = urllib3.PoolManager().request("GET", f"http://oracc.museum.upenn.edu/{f}/{member}/xml").data
                    xml_file.write(Data)
            except urllib3.exceptions.MaxRetryError:
                continue


if __name__ == "__main__":
    # download_project_list(True)
    read_catalogs()

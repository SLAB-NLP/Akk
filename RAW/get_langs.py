import pandas as pd
import json
import glob


def get_catalog(lst: list, num: int = 5):
    if num == 0:
        return lst
    lst += glob.glob(f"jsons_unzipped/{'*/'*num}/cata*.json", recursive=True)
    return get_catalog(lst, num-1)


cats = get_catalog([], 7)
cols = [c.split('\\')[-2] for c in cats]
periods = []
langs = []


def get_period(elem: dict):
    return f'{elem["period"]}' if elem.get("period") else ""


def get_language(elem: dict):
    return f'{elem["language"]}' if elem.get("language") else ""


def files_write(langs: list, period: list):
    with open("data/lists/periob ds.txt", "w", encoding="utf_8") as file:
        file.writelines(period)
    with open("data/lists/langhjs.txt", "w", encoding="utf_8") as file:
        file.writelines(langs)


def create_lists(cats: list):
    global langs, periods
    for c in cats:
        try:
            with open(c, "r", encoding="utf_8") as file:
                data = json.load(file)["members"]
            for d in data:
                periods.append(get_period(data[d]))
                langs.append(get_language(data[d]))
        except:
            continue
    langs = list(set(langs))
    periods = list(set(periods))


def count_langs(lang: str):
    lag = {}
    lag["Name"] = lang
    global cats
    # global l
    global cols
    for c, projects in zip(cats, cols):
        counter = 0
        try:
            with open(c, "r", encoding="utf_8") as file:
                data = json.load(file)["members"]
            for d in data:
                counter += 1 if get_language(data[d]) == lang else 0
            lag[projects] = counter
        except:
            continue

    return lag


if __name__ == "__main__":
    create_lists(cats)
    df = pd.DataFrame()
    for l in langs:
        try:
            namen = count_langs(l)
            if namen["Name"] == "":
                namen.update({"Name": "None"})
            df = df.append(namen, True)
        except ValueError:
            continue
    with pd.ExcelWriter("data/lists/___excel.file.xlsx", "openpyxl") as xl:
        df.to_excel(xl, index=False)

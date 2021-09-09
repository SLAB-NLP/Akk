import json, glob


def get_catalog(lst: list, num: int = 5):
    if num == 0:
        return lst
    lst += glob.glob(f"jsons_unzipped/{'*/'*num}/cata*.json", recursive=True)
    return get_catalog(lst, num-1)
def get_period(elem: dict):
    return f'{elem["period"]}\n' if elem.get("period") else ""
def get_language(elem: dict):
    return f'{elem["language"]}\n' if elem.get("language") else ""
def files_write(langs: list, period: list):
    with open("data/lists/periob ds.txt","w",encoding="utf_8") as file:
        file.writelines(period)
    with open("data/lists/langhjs.txt","w",encoding="utf_8") as file:
        file.writelines(langs)
        
if __name__ == "__main__":
    files = get_catalog([], 5)
    langs = []
    periods = []
    for f in files:
        with open(f'{f}',encoding="utf_8") as filus:
            elems = json.load(filus)
        elems = elems["members"] if elems.get("members") else {}
        for m in elems:
            langs.append(get_language(elems[m]))
            periods.append(get_period(elems[m]))
    langs = list(set(langs))
    periods = list(set(periods))
    files_write(langs,periods)

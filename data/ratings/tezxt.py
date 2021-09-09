with open(r"D:\Drive\לימודים\מאגרי מידע\זמני\ancient-text-processing\jsons_unzipped\saao\saa01\catalogue.json","r",encoding="utf_8") as file:
    catalog = eval(file.read())["members"]
    
rulers = []

for c in catalog:
    cat = catalog[c]
    if cat["period"] == "Neo-Assyrian" and cat.get("ruler"):
        rulers += cat["ruler"]

TIME_TEXT = {

             "Neo-Assyrian":
             ["Adad-nerari II","Tukulti-Ninurta II","Ashurnasirpal II","Shalmaneser III","Šamši-Adad V","Shammuramat",
              "Adad-nerari III","Shalmaneser IV","Aššur-dan III","Aššur-nerari V","Tiglath-Pileser III","Shalmaneser V","Sargon II",
              "Sennacherib","Esarhaddon","Ashurbanipal","Aššur-etel-ilani","Sîn-šarru-iškun","Sin-shumu-lishir","Ashur-uballit II"],
             "Middle Assyrian":
             ['Aššur-uballiṭ I', 'Enlil-nārārī', 'Arik-dīn-ili',
                              'Sargon I', 'Puzur-Aššur II', 'Narām-Sîn', 'Erišum II'
                              'Adad-nārārī I',
                              'Shalmaneser I',
                              'Tukultī-Ninurta I',
                              'Aššur-nādin-apli',
                              'Enlil-kudurrī-uṣur',
                              'Aššur-nārārī III',
                              'Ninurta-apil-Ekur',
                              'Aššur-dān I',
                              'Ninurta-tukultī-Aššur',
                              'Mutakkil-Nusku',
                              'Aššur-rēša-iši I'],
             "Old Assyrian": 
                            ["Aminu", "Ilā-kabkabī", "Yaškur-ilu", "Yakmenu", "Yakmisu", "Ilī-Wēr", "Ḫayānu", "Samānu", "Ḫallî", 
                             "Apiašal", "Sulilu", "Kikkiya", "Akiya", "Puzur-Aššur (I)", "Šalim-aḫum", "Ilu-šūma", "Erišum (I)", 
                             "Ikūnum", "Sargon (I)", "Puzur-Aššur (II)", "Narām-Sîn", "Erišum (II)", "Samsī-Addu (I)"]
             }

with open('data/lists/Text Origin.txt', 'r', encoding='utf_8') as file:
    TEXT_ORIGIN = file.read()

def _names_lines_ratio (file: str):
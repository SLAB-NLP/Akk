import urllib3
import bs4

ORACC_PROJECT_NAME = ""

SUPERSCRIPTS_TO_UNICODE_CHARS = {
    "<super>1</super>": "\U0001F600",
    "<super>m</super>": "\U0001F600",
    "<super>d</super>": "\U0001F601",
    "<super>f</super>": "\U0001F602",
    "<super>ki</super>": "\U0001F603",
    "<super>kur</super>": "\U0001F604",
    "<super>giš</super>": "\U0001F605",
    "<super>uru</super>": "\U0001F606",
    "<sup>1</sup>": "\U0001F600",
    "<sup>m</sup>": "\U0001F600",
    "<sup>d</sup>": "\U0001F601",
    "<sup>f</sup>": "\U0001F602",
    "<sup>ki</sup>": "\U0001F603",
    "<sup>kur</sup>": "\U0001F604",
    "<sup>giš</sup>": "\U0001F605",
    "<sup>uru</sup>": "\U0001F606",
    "<sup>V</sup>": "",
}


def _has_project(link: str) -> bool:
    """HELPER: check if the link in oracc, and has a project 

    :param link: an oracc project link
    :type link: str
    :return: true if the link is in oracc with project, false otherwise
    :rtype: bool
    """
    return ("oracc.org" in link.lower() or "oracc.museum.upenn.edu" in link.lower()) and (
                "projectlist" not in link.lower())


# TODO: check the edge cases


def get_data(link: str, data: str) -> str:
    """this function scraps the project's large html file to extract the data from it

    :param link: the project main link
    :type link: str
    :param data: data to extract, akkadian or objects. if objects, remember to remove the qpn-x- from the title
    :type data: str
    :return: the data of the website, as string
    :rtype: str
    """
    if "akk" in data:
        pass
    elif data != "qpn":
        data = f"qpn-x-{data}"
    else:
        pass
    data_from_oracc = urllib3.PoolManager().request("GET", f"{link}/cbd/{data}/onebigfile.html").data.decode()
    if _has_project(link) and "404 Not Found" not in data_from_oracc:
        global ORACC_PROJECT_NAME
        ORACC_PROJECT_NAME = link[link.rfind("/"):]
        return data_from_oracc


def _divide_links(data: str, class_type="body") -> list:
    """HELPER: divides the link to a list of classes. the default is 'body'

    :param data: the raw data from the oracc website
    :type data: str
    :param class_type: class of the divided links, defaults to "body"
    :type class_type: str, optional
    :return: list of tags
    :rtype: list
    """
    data = bs4.BeautifulSoup(data, "html.parser")
    return [divs for divs in data.select(f'div[class^="{class_type}"]')]


def _get_key_words(data: bs4.element.Tag) -> str:
    return data.select('span[class ^="cf "]')[0].string


def _divide_writings(data: bs4.element.Tag):
    return data.select("span[class^='w']")


def _list_writings(data: list) -> list:
    write_list = []
    for write_varients in data:
        for divided in _divide_writings(write_varients):
            write_list.append(_combine_writing(divided))
    return list(set(write_list))


def _super_to_dot(superscript: str) -> str:
    """HELPER: converts the superscript in oracc to unicode

    :param superscript: the superscript classifier
    :type superscript: str
    :return: replacement of the classifier in unicode
    :rtype: emoji char
    """
    return SUPERSCRIPTS_TO_UNICODE_CHARS[superscript]


def _combine_writing(contents_data: bs4.element.Tag) -> str:
    """HELPER: gets as input a list of tags and returns the text of the tags

    :param contents_data: bs4 tag list of the data
    :type contents_data: list
    :return: the text
    :rtype: str
    """
    writing = ""
    for content in contents_data.contents:
        if str(content) in SUPERSCRIPTS_TO_UNICODE_CHARS:
            writing += _super_to_dot(str(content).replace("v", ""))
        elif content.string is None:
            writing += content.text.replace("v", "")
        else:
            writing += content.string.replace("v", "")
    return writing


def dict_words_writings(divided_links: list) -> dict:
    """converts the data from oracc you got from the 'get_data' function
    to a dictionary of 

    :param divided_links: [description]
    :type divided_links: list
    :return: [description]
    :rtype: [type]
    """
    dict_words_and_writings = {}
    for DividedLink in divided_links:
        dict_words_and_writings[_get_key_words(DividedLink)] = _list_writings(DividedLink)
    return dict_words_and_writings


if __name__ == '__main__':
    s = dict_words_writings(_divide_links(get_data("http://oracc.museum.upenn.edu/saao/", "qpn")))
    with open('t.txt', 'w', encoding='utf_8') as file:
        file.write(str(s))
    print("yes, m'lord")

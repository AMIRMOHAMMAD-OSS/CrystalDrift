"""Read the public Drive directory used by the authors to publish their weights."""
import sys

import requests
from bs4 import BeautifulSoup

for folder in sys.argv[1:]:
    r = requests.get("https://drive.google.com/drive/folders/" + folder, timeout=40)
    r.raise_for_status()
    nodes = BeautifulSoup(r.text,"html.parser").select("[data-id]")
    for n in nodes:
        print(n.get("data-id"), ascii(n.get("aria-label", "") or n.text[:140]))

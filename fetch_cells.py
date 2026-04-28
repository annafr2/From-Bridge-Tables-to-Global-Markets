import urllib.request
from bs4 import BeautifulSoup

url = 'http://db.eurobridge.org/Repository/competitions/22Madeira/microsite/Asp/BoardDetails.asp?qmatchid=97024'
html = urllib.request.urlopen(url).read().decode('iso-8859-1')
soup = BeautifulSoup(html, 'html.parser')
highlight_table = soup.find('table', id='highlight')
cells = highlight_table.find_all('tr', recursive=False)[1].find_all('td', recursive=False)
with open('scratch_cells.txt', 'w', encoding='utf-8') as f:
    for i, cell in enumerate(cells):
        f.write(f'Cell {i}: {cell}\n')

url_herning = 'http://db.eurobridge.org/Repository/competitions/24Herning/microsite/Asp/BoardDetails.asp?qmatchid=138742'
html_h = urllib.request.urlopen(url_herning).read().decode('iso-8859-1')
soup_h = BeautifulSoup(html_h, 'html.parser')
highlight_table_h = soup_h.find('table', id='highlight')
cells_h = highlight_table_h.find_all('tr', recursive=False)[1].find_all('td', recursive=False)
with open('scratch_cells_h.txt', 'w', encoding='utf-8') as f:
    for i, cell in enumerate(cells_h):
        f.write(f'Cell {i}: {cell}\n')

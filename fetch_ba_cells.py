import urllib.request
from bs4 import BeautifulSoup

url_ba = 'http://db.eurobridge.org/Repository/competitions/22Madeira/microsite/Asp/BoardAcross.asp?qboard=001.01..2220'
html = urllib.request.urlopen(url_ba).read().decode('iso-8859-1')
soup = BeautifulSoup(html, 'html.parser')

has_tooltip = soup.find('a', class_='tooltip') is not None
print('BoardAcross has tooltip:', has_tooltip)

# Let's see the first open room row in BoardAcross for Madeira
results_table = None
for tbl in soup.find_all("table"):
    first_tr = tbl.find("tr", recursive=False)
    if not first_tr: continue
    first_tds = first_tr.find_all("td", recursive=False)
    if first_tds and "Table" in first_tds[0].get_text():
        results_table = tbl
        break

if results_table:
    rows = results_table.find_all('tr', recursive=False)
    cells = rows[1].find_all('td', recursive=False)
    with open('scratch_ba_cells.txt', 'w', encoding='utf-8') as f:
        for i, c in enumerate(cells):
            f.write(f'BA Cell {i}: {c}\n')

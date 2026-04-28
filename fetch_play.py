import urllib.request
from bs4 import BeautifulSoup

url = 'http://db.eurobridge.org/Repository/competitions/22Madeira/microsite/Asp/PlayDetails.asp?qboard=001.01..2220&qtournid=2220&qmatchid=97024'
html = urllib.request.urlopen(url).read().decode('iso-8859-1')
soup = BeautifulSoup(html, 'html.parser')

has_bidding = False
for txt in soup.stripped_strings:
    if 'Pass' in txt or '1NT' in txt:
        has_bidding = True

with open('scratch_play.html', 'w', encoding='utf-8') as f:
    f.write(soup.prettify())

print('PlayDetails has typical bidding string:', has_bidding)

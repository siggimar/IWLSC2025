'''
Script to plot location of projects from where CPTu data is collected.
    relies on classes from map_norway.py and map_sosi_files.py, written by S. M. Valsson
    
    map data on SOSI format included, taken from: https://kartkatalog.geonorge.no/
    which is distributed under the Norwegian NLOD licence for open use of public data.
'''

from map_norway import map_norway

projects = { # coordinates of parcels selected ca. in center
    'E16 Kongsvinger':[ 333941, 6676124 ],
    'E6 Helgeland Sør':[ 424264, 7244985 ],
    'E6 Høybukta-Hesseng':[ 1069066, 7804190 ],
    'E6 Klett':[ 264941, 7029949 ],
    'E6 Megården - Mørsvikbotn':[ 528302, 7492423 ],
    'E6 Sørfoldtunnelene':[ 524441, 7481957 ], # approx
    'E6 Sveningelv-Valryggen':[ 426333, 7268753 ],
    'E6 Svenningelv-Lien':[ 423790, 7276012 ],
    'E6 Transfarelv bru':[ 822558, 7785462 ],
    'E6 Valryggen-Lien':[ 425252, 7275931 ],
    'Førdepakken tiltak 1: Angedalsvegen–Hafstadvegen':[ 14308, 6847156 ],
    'Fv 64 Bruvollbrua':[ 128173, 7012800 ],
    'Fv 670 Todalsfjordprosjektet':[ 170695, 6982923 ],
    'Fv 715 Rørvik ferjekai':[ 258336, 7051556 ],
    'FV 717 Sund - Bradden':[ 248877, 7059431 ],
    'Fv. 14 Berfjord':[ 268112, 7122316 ],
    'Fv. 36 Gråbrekkmølle':[ 295538, 7046362 ],
    'Fv. 64 Bådalen-Vebenstad':[ 125038, 7008367 ],
    'Fv. 705 Bjørgmyra-Frigården':[ 296925, 7040325 ],
    'Fv. 715 Urdskaret - Hestmoen':[ 289056, 7130957 ],
    'Fv. 98 Torhop-Smalfjordbotn':[ 983479, 7868073 ],
    'Fv.17 Østvik-Beitstadsundet':[ 321116, 7112938 ],
    'Fv.416 Risørveien':[ 162941, 6522948 ],
    'Fv.848 Ibestadtunnelen - midlertidig fergeleie':[ 588734, 7632756 ],
    'Fv12 Mjølan rundkjøring':[ 462328, 7356719 ],
    'Fv17 Holm fergeleie':[ 364911, 7231893 ],
    'Hålogalandsvegen':[ 531416, 7602187 ],
    'Hålogalandsvegen, parsell 15':[ 531735, 7610490 ],
    'Kjøpsvik fergeleie':[ 556536, 7553872 ],
    'Nordstrandveien':[ 474046, 7464788 ],
    'RV 80 Vestmyra-Klungset':[ 513195, 7461296 ],
    'Rv.710 Ingdal-Valset':[ 242903, 7056246 ],
    'Rv22 Rudskogen-E18':[ 295090, 6608192 ],
    'Segelstein':[ 503522, 7509638 ],
    'Skjeggestadbrua kollaps 2015':[ 233739, 6601866 ],
    'Testfelt Kjellstad':[ 233793, 6635207 ],
    'Fv40 og Fv221 Svarstad':[ 213731, 6595746 ],
    'Fv900 Holmestrand sentrum':[ 234800, 6603866 ],
}


if __name__=='__main__':
    simple_map = map_norway( reload_map=True, draw_elements = {'regions':True, 'curves':True, 'places':True} )
    
    for project in projects:
        if projects[project]:
            x, y = projects[project][0], projects[project][1]
            simple_map.ax.plot( x, y, marker='o', ms=12, mec=(0,0,0), mfc=(0,0,1), zorder=999 )

    simple_map.show()
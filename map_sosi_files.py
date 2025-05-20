'''
script used to generate mep for Figure 1.
imported indirectly to _Figure_1_map.py



quick n'dirty SOSI parser to generate map of Norway
'''

import re






class sosi_file():
    def __init__( self, file_path ):
        self.file_path = file_path
        self.file_contents = ''

        self.info = {}
        self.curves = {}
        self.regions = {}
        self.names = {}

        self.strings = {}

        self.plot_styles = { # colors as (r,g,b)/(r,g,b,a): values: 0-255 (incl. alpha)

            #regions
            'region_standard': {'zorder':1, 'lw': 0.05, 'ec': (0,0,0,200), 'fc':(255,255,255), 'ls':'-'}, # always
            'Kommune':  {'fc': (255, 255, 255, 0 ), 'ls':'--' }, # overwrites standard definition
            'Tettbebyggelse': {'fc': (220, 230, 230) },
            'Industriområde': {'fc': (220, 230, 230) },
            'Myr': {'fc': (230, 190, 190) },
            'ÅpentOmråde': {'fc': (241, 236, 226) },
            'Havflate': {'zorder':0,'fc': (209, 234, 250) },
            'Innsjø': {'fc': (178, 216, 255) },
            'ElvBekk': {'fc': (178, 216, 255) },
            'Skog': {'fc': (214, 232, 208) },
            'SnøIsbre': {'fc': (253, 253, 253) },
            'Naturvernområde': {'fc': (255, 0, 0,50) },


            # text:  fc & ec apply to marker
            'name_standard': {'zorder': 999, 'c': (100, 100, 100), 'size': 10, 'txt_offset': 22000, 'fc': ( 255, 0, 0 ), 'ec': ( 0, 0, 0 ), 'mew':2, 'ms':18 }, #'fc': ( 250, 250, 250 ), 'ec': ( 120, 120, 120 )

            
            # curves
            'Kystkontur': {'zorder': 2, 'c': (100,100,100), 'lw': 0.25},
            'Innsjøkant': {'zorder': 2, 'c': (100,100,100), 'lw': 0.25},
            'Riksgrense':{'zorder': 90, 'c': (250, 150, 150), 'lw': 1},
            'Grunnlinje':{'zorder': 2, 'c': (46, 150, 200), 'lw': 1},

            # roads/railway: drop ferries
            'road_standard': {'zorder': 5, 'c': (200, 200, 200), 'lw': 1, 'bc': (100,100,100), 'bz_order': 3, 'd_blw':1.5},
            'Bane': {'zorder': 5, 'c': (200, 200, 200), 'lw': 1}, # railway standard
            'Ev': {'zorder': 8, 'c': (250, 220, 150), 'lw': 1},
            'Rv': {'zorder': 7, 'c': (250, 150, 150), 'lw': .5},
            'Fv': {'zorder': 6, 'c': (250, 210, 210), 'lw': .5},            
            'freeway': { 'lw': 1.5 }
        }

        self.scale_colors()
        self.read_file()
        self.parse_sosi()


    def scale_colors( self ):
        for h in self.plot_styles:
            for item in self.plot_styles[h]:
                if isinstance(self.plot_styles[h][item], tuple ): # color definition
                    self.plot_styles[h][item] = tuple( [k/255 for k in self.plot_styles[h][item]] )


    def read_file( self ):
        with open( self.file_path, 'rb' ) as f:
            f_contents = f.read()
        self.file_contents = f_contents.decode( 'utf-8-sig' ) # contains utf-8 with BOM


    def parse_sosi( self ):
        self.calc_data_blocks()
        self.parse_blocks()
        self.region_coords()


    def calc_data_blocks( self ):
        pattern = r'^\.[A-Z]'
        lines = self.file_contents.split( '\r\n' )

        self.blocks = []
        current_block = []

        for line in lines:
            if re.match( pattern, line ):
                if current_block: self.blocks.append( current_block ) # keep last block
                current_block = [line] # start new
            elif current_block:
                current_block.append( line ) # continue with block

        if current_block: self.blocks.append( current_block ) # keep last block


    def parse_blocks( self ):
        for block in self.blocks:
            id_split = block[0][1:].split( ' ' )

            if id_split[0] == 'KURVE':
                curve_id = int( id_split[1][:-1] )
                curve_type = block[1][2:].split( ' ' )[1]

                res = self.curve_geometry( block )
                self.curve_attributes( res, block )
                res['curve_type'] = curve_type
                self.curves[curve_id] = res

            elif id_split[0] == 'FLATE':
                region_id = int( id_split[1][:-1] )
                res = self.region_def( block )
                self.regions[region_id] = res

            elif id_split[0] == 'TEKST':
                name_id = int( id_split[1][:-1] )
                res = self.text_def( block )
                self.names[name_id] = res

            else:
                pass # nothing for now


    def curve_attributes( self, res, block ):
        all_lines = '\n'.join( block )
        
        if 'Samferdsel_SOSI.sos' in self.file_path:
            a=1
        c_attributes = {'type': '..OBJTYPE ', 'road_nr': '..VNR ', 'freeway_type': '..MOTORVEGTYPE ', 'medium': '..MEDIUM ', 'name': '..NAVN '}
        for key, value in c_attributes.items():
            self.insert_value_by_key( res, key, all_lines, value )



        if False: # investigate available data
            ignore = ['FiktivDelelinje', 'VegSenterlinje', 'Bilferjestrekning',\
                    'AnnenBåtrute','ElvBekk','Naturverngrense','Dataavgrensning',\
                    'Arealbrukgrense', 'Kommunegrense', 'Fylkesgrense','Territorialgrense','Grunnlinje']
            if res['type'] not in self.plot_styles and res['type'] not in ignore:

                print( res['type'] )
                a=1
        

        if res['type'] in self.plot_styles:
            res.update( self.plot_styles[res['type']] ) # all curves

        if 'road_nr' in res:
            res.update( self.plot_styles['road_standard'] ) # all roads
            r_name, type_name = self.road_name_type( res['road_nr'] )
            res['road_name'] = r_name
            if type_name in self.plot_styles:
                res.update( self.plot_styles[type_name] )


            if 'freeway_type' in res:
                if res['freeway_type']=='Motorveg':
                    res.update( self.plot_styles['freeway'] )

        a=1

    def road_name_type( self, raw_name ):
        r_name = raw_name.replace(' ', '').replace('*', 'V').replace('V','v')

        type_name = ''.join([c for c in r_name if not c.isdigit()])
        return r_name, type_name



    def text_def( self, block ):
        all_lines = '\n'.join( block )

        res = self.curve_geometry( block )

        txt_attributes = {'type': '..OBJTYPE ', 'lng': '..SPRÅK ', 'fulltext': '..FULLTEKST ',
                 'text': '..STRENG ', 'text_type': '..SKRIVEMÅTENUMMER ', 'main_group': '..NAVNEOBJEKTHOVEDGRUPPE ',
                 'sub_group': '..NAVNEOBJEKTGRUPPE ', 'main_type': '..NAVNEOBJEKTTYPE '}

        for key, value in txt_attributes.items():
            self.insert_value_by_key( res, key, all_lines, value )


        res.update( self.plot_styles['name_standard'] )
        if res['type'] in self.plot_styles:
            res.update( self.plot_styles[res['type']] )
        
        res['txt_def'] = [res['text'], res['coords'][0][0], res['coords'][1][0] ]
        
        return res


    def insert_value_by_key( self, res, res_key, some_str, str_key ):
        if str_key in some_str:
            res[res_key] = some_str.split(str_key)[1].split('\n')[0]


    def region_coords( self ):
        for region in self.regions:
            self.regions[region]['coords'] = [ [], [] ]
            refs = self.regions[region]['refs']

            for ref in refs:
                ref_coords = self.curve_coords( ref ) # reverses if needed
                self.regions[region]['coords'][0] += ref_coords[0]
                self.regions[region]['coords'][1] += ref_coords[1]


    def curve_coords( self, ref ):
        backwards = ref<0
        i_ref = abs( ref )

        x, y = self.curves[i_ref]['coords'][0], self.curves[i_ref]['coords'][1]
        if 'ref_coord' in self.curves[i_ref]:
            x = self.curves[i_ref]['ref_coord'][0] + x
            y = self.curves[i_ref]['ref_coord'][1] + y

        if backwards:
            x.reverse()
            y.reverse()
        return [ x, y ]


    def curve_geometry( self, block ):
        coord_components = '\n'.join(block).split( '..NØ\n' )

        n=len( coord_components )

        res = {}

        for i in range(1, n ): # definitions in idx_0
            tmp_str = coord_components[i].replace( '...KP 1', '' )
            tmp_str.strip()

            x, y = [], []

            lines = tmp_str.split( '\n' )
            for line in lines:
                if len(line)>0:
                    parts = line.strip().split( ' ' )
                    y.append( float( parts[0] )/100 )
                    x.append( float( parts[1] )/100 )

            if i==1 and i<(n-1):
                res['ref_coord']=[x,y]
            else: 
                res['coords']=[x,y]
            
        return res


    def region_def( self, block ): # only for administrative regions
        all_lines = '\n'.join( block )
        res = {}
        
        is_municipality = '..OBJTYPE Kommune' in all_lines

        res['type']=block[1][2:].split( ' ' )[1]
        if is_municipality: 
            res['muni_nr']=block[2][2:].split( ' ' )[1]
            res['name']=block[3][2:].split( ' ' )[1]
            res['date']=block[4][2:].split( ' ' )[1]

        # finally curve refs
        res['refs'] = []
        res['par_refs'] = []
        refs_string = all_lines.split( '..REF' )[1].split( '..NØ' )[0]
        par_refs_string = '' # refs in parenthesis

        if '(' in refs_string: # regulated water?
            # prep special refs
            idx = refs_string.find('(') # first index of openinig parenthesis in ref list
            par_refs_string = refs_string[idx:] # everything after idx
            par_refs_string = par_refs_string.replace('(','').replace(')','').replace( '\n', '' )
            refs_string = refs_string[:idx] # also remove the above from standard definition

        refs_string = refs_string.replace( '\n', '' )

        refs = refs_string.split( ':' )
        par_refs = par_refs_string.split( ':' )

        for i in range( 1, len(refs) ): # standard refs
            res['refs'].append( int(refs[i]) )
        
        for i in range( 1, len(par_refs)): # refs in parenthesis
            res['par_refs'].append( int(par_refs[i]))

        # add plot-styles
        res.update( self.plot_styles['region_standard'] )
        if res['type'] in self.plot_styles:
            res.update( self.plot_styles[res['type']] )
        else:
            pass
            #self.print_once( res['type'] )

        return res
    
    def print_once( self, some_str ):
        tmp_hash = hash( some_str )
        if not tmp_hash in self.strings:
            self.strings[tmp_hash] = some_str
            print( some_str )
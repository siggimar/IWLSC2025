'''
script to generate map used in Figure 1.
imprted in _Figure_1_map.py




Maps downloaded from Kartverket:  https://kartkatalog.geonorge.no/metadata/kartverket/n5000-kartdata/c777d53d-8916-4d9d-bae4-6d5140e0c569

utm-33
filetype:  *.sos
format version: SOSI 4.5
coordinates stored as cm

running this module loops through available maps and saves map caches
  requires substantial space, but speeds load times significantly
'''


import matplotlib.pyplot as plt
import matplotlib.patheffects as pe # for text shadows
import matplotlib.ticker as ticker
from map_sosi_files import sosi_file
import pickle
import os


class map_norway():
    """Class returns a map object for Norway.  
    
Examples:
        >>> standard_map = map_norway() # low detail
        >>> standard_map.show()
        ...
        >>> coastline_map = map_norway( map_type='simplified' ) # just coastline + borders
        >>> coastline_map.show()
        ...
        >>> detailed_map = map_norway( detail='N1000' )
        >>> detailed_map.show()
        ...
        >>> detailed_coastline_map = map_norway(map_type='simplified', detail='N1000')
        >>> detailed_coastline_map.show()
        ... 

        the coordinate system can be changed to utm-32 as:
        >>> standard_map = map_norway( coord_system='utm-32' )
        >>> standard_map.show()

        This class does not provide a detailed map in utm-32

    """
    def __init__( # standard values for optional parameters
            self,
            map_type='full',
            coord_system='utm-33',
            detail='N5000',
            reload_map=False,
            draw_elements={'regions':True,'curves':True,'places':True}
        ):
        save_filename = 'saved_map'
        extension = '.pkl'
        self.roots = {
            'N5000/utm-32' : 'map_norway/N5000/utm-32',
            'N5000/utm-33' : 'map_norway/N5000/utm-33',
            'N1000/utm-33' : 'map_norway/N1000/utm-33'
        }
        self.map_types = ['simplified', 'full']
        self.map_type = map_type if map_type in self.map_types else self.map_types[0]
        self.coord_system = coord_system

        self.reload = reload_map
        self.draw_elements = draw_elements
 
        root_id = detail + '/' + coord_system
        if root_id  not in self.roots: coord_system = 'N5000/utm-33'
        self.root = self.roots[ root_id ]
        
        self.save_path = os.path.join( self.root, save_filename + '-' + self.map_type + extension )
        self.get_map()

        if not self.loaded_from_save: self.save_map()


    def get_map( self ):  
        self.loaded_from_save = False
        if os.path.isfile( self.save_path ) and not self.reload:
            # load saved figure
            self.load_saved_map()
            self.loaded_from_save = True
        else:
            # parse files and draw figure
            self.files = []
            self.load_map_data() # reads .sos files

            if self.map_type==self.map_types[0]:
                self.simple_map()
            else:
                self.full_map()

            self.files = []



    def add_points( self, x, y, s=None, c=None, marker=None, zorder=100 ):
        self.ax.scatter( x, y, s=s, c=c, marker=marker, zorder=zorder )


    def add_curve( self, x, y, c=(0,0,0), lw=0.2, ls='-', zorder=100 ):
        self.ax.plot( x, y, c=c, lw=lw, ls=ls, zorder=zorder )


    def add_labeled_point_set( self, X, labels ):
        pass


    def show( self, fullscreen=True, grid=False ):
        if fullscreen:
            mng = plt.get_current_fig_manager()
            mng.window.state('zoomed')
        self.ax.grid(grid)
        plt.show()


    def simple_map( self ):
        regions = [] # these types will be drawn (if present in .sos files)
        curves  = ['Kystkontur', 'Riksgrense']
        places  = [ ]
        return self.draw_map( region_list=regions, curve_list=curves, place_list=places )


    def full_map( self ):
        regions = ['ÅpentOmråde','Havflate','Innsjø','Skog','SnøIsbre','Tettbebyggelse','Myr','ElvBekk', 'Industriområde']
        curves  = ['Kystkontur', 'Innsjøkant', 'Riksgrense', 'VegSenterlinje']
        places  = [ 'Oslo', 'Stavanger', 'Bergen', 'Trondheim', 'Tromsø', 'Bodø'] #, 'Stjørdal' ]
        return self.draw_map( region_list=regions, curve_list=curves, place_list=places )


    def draw_map( self, region_list=[], curve_list=[], place_list=[] ):
        self.fig, self.ax = plt.subplots( tight_layout=True )
        self.draw_regions( region_list )
        self.draw_selected_places( place_list )
        self.draw_curves( curve_list )
        
        self.format_axis()
        

    def draw_regions( self, region_list ):
        if self.draw_elements['regions']:
            for file in self.files:
                for region in file.regions:
                    if file.regions[region]['type'] in region_list:
                        x, y = file.regions[region]['coords']
                        self.ax.fill( x, y, fc=file.regions[region]['fc'], ec=file.regions[region]['ec'], \
                                lw=file.regions[region]['lw'], ls=file.regions[region]['ls'],\
                                zorder=file.regions[region]['zorder'] )


    def draw_selected_places( self, name_list ):
        if self.draw_elements['places']:
            for file in self.files:
                for name in file.names:                
                    if file.names[name]['fulltext'] in name_list:
                        t, x, y = file.names[name]['txt_def']
                        self.ax.text( x, y, t+'\n', size=file.names[name]['size'], c=file.names[name]['c'], \
                                va='bottom', ha='center', path_effects=[pe.withStroke(linewidth=2, foreground=(1,1,1))], zorder=file.names[name]['zorder'])
                        self.ax.plot( x, y, 
                                    markerfacecolor=file.names[name]['fc'], 
                                    markeredgecolor=file.names[name]['ec'] ,
                                    marker='o', ms=file.names[name]['ms'], 
                                    mew=file.names[name]['mew'],
                                    zorder=file.names[name]['zorder'] )


    def draw_curves( self, curve_list ):
        if self.draw_elements['curves']:
            for file in self.files:
                for curve in file.curves:
                    if file.curves[curve]['type'] in curve_list:
                        x, y = file.curve_coords(curve)

                        if file.curves[curve]['type']  == 'VegSenterlinje': # roads get boundary on lines
                            self.ax.plot( x, y, c=file.curves[curve]['bc'], lw=file.curves[curve]['lw']*file.curves[curve]['d_blw'],\
                                    zorder=file.curves[curve]['bz_order'] )
                        
                        self.ax.plot( x, y, c=file.curves[curve]['c'], lw=file.curves[curve]['lw'], zorder=file.curves[curve]['zorder'] )


    def format_axis( self ):
        self.ax.axis('equal')

        # remove scienific notation
        self.ax.xaxis.set_major_formatter('{x:1.0f}')
        self.ax.yaxis.set_major_formatter('{x:1.0f}')

        # add labels
        self.ax.set_xlabel( self.coord_system.upper().replace('-','') + ' - Easting')
        self.ax.set_ylabel( self.coord_system.upper().replace('-','') + ' - Northing')

        # tick increments
        base_increment = 200000
        self.ax.xaxis.set_major_locator(ticker.MultipleLocator(base=base_increment))
        self.ax.yaxis.set_major_locator(ticker.MultipleLocator(base=base_increment))


    def get_ax( self ):
        return self.ax
    def get_fig( self ):
        return self.fig


    def load_map_data( self ):
        sosi_file_paths = self.map_files()
        for sosi_file_path in sosi_file_paths:
            self.files.append( sosi_file( sosi_file_path ) )


    def map_files( self, extension='.sos' ):
        ext = extension.lower()

        files = os.listdir( self.root ) # all files
        if extension is not None:
            files = [ os.path.join( self.root, f ) for f in files if ext in f.lower()]

        return files


    def load_saved_map( self ):
        with open( self.save_path, 'rb' ) as f:
            self.fig, self.ax = pickle.load( f )


    def save_map( self ):
        with open( self.save_path, 'wb' ) as f:
            pickle.dump( [self.fig, self.ax], f )


def rebuild_caches():
    maps = {
        'utm-33':{'map_type':['simplified', 'full' ],'detail':['N5000', 'N1000']},
        'utm-32':{'map_type':['simplified', 'full' ],'detail':['N5000']}
    }

    import time
    print('Rebuilding map cache')
    start = time.time()
    i=1
    for crd_sys, definition in maps.items():
        for map_type in definition['map_type']:
            for detail_level in definition['detail']:
                print('   working on map: ' + str(i), end='\r')
                my_map = map_norway( map_type=map_type, coord_system=crd_sys, detail=detail_level, reload_map=True)
                i += 1
    print('\n\nFinished in: ' + str(round(time.time()-start,2)) + 's')

def quick_map():



    coastline_map = map_norway( map_type='simplified' ) # just coastline + borders
    coastline_map.show()

    #detailed_coastline_map = map_norway(map_type='simplified', detail='N1000')
    #detailed_coastline_map.show()

    #standard_map = map_norway() # low detail
    #standard_map.show()

    #detailed_map = map_norway( detail='N1000', reload_map=True )
    #detailed_map.show()
    
    
    #my_map = map_norway( reload_map=True )
    #my_map.show( grid=True )
    

if __name__ == '__main__':
    #rebuild_caches()
    quick_map()

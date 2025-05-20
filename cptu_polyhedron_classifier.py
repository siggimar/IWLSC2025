import numpy as np
import os
import pyvista as pv
import vtk # for raw strings in axis labels
from plyfile import PlyData


'''
    Class to load, use, and visualize triangulated polyhedron classification models for CPTu.

    Prepared as supplement to paper:
    Valsson, S. M., Degago, S. A., Eiksund, G. R., (2025) Triangulated density models for detecting brittle materials with CPTu. 
    submitted to the 3rd IWLSC conference, Quebec. 

    Adapted from earlier own Excel VBA implementation: https://doi.org/10.5281/zenodo.14853253
'''


class polyhedron_clf():
    def __init__( self, folder='', label_outside=-1 ):
        self.polyhedrons = {}
        self.label_outside =label_outside

        if len(folder)>0: 
            self.load_model( folder )
            self.optimize_order()
            self.set_lims() # for quick bbox check

        self.label_dict = {
            'Bq': '\n' + r'$B_q$',
            'Fr': '\n' + r'$log_{10} \left( F_r \right)$',
            'Qt': '\n' + r'$log_{10} \left( Q_t \right)$',
        }

        self.ax_lims = {
            'Bq': [-0.6, 2 ],
            'Fr': [ 0.1, 10  ],
            'Qt': [ 1, 1000  ],
        }


    def load_model( self, folder ):
        files = self.files_in_folder( folder, filetype='.ply' )

        for file in files:
            ply_data = PlyData.read( file )
            comments = self.read_comments( ply_data.comments )
            
            # attributes for presentation && introduction
            for attribute in [ 'software', 'model_name', 'variables', 'axis_scales' ]:
                if attribute in comments:
                    setattr( self, attribute, comments[attribute] )

            self.parse_vars()
            self.logs = [ some_scale.lower() == 'log' for some_scale in comments['axis_scales'].split(' ') ]

            # define polyhedrons
            self.polyhedrons[ int(comments['value']) ] = polyhedron_clf.polyhedron( ply_data, comments )


    def optimize_order( self ):
        def construct_ordered_list(some_list, element):
            # calculate distances from element to every other element
            distances = [(abs(element - x), x) for x in some_list]
            distances.sort()

            return [x for _, x in distances] # return ordered elements
        
        ids = list( self.polyhedrons.keys() )
        self.order = { some_id:ids for some_id in ids } # basic order

        for key in self.order.keys():
            self.order[key] = construct_ordered_list( ids, key )

        self.order[ self.label_outside ] = self.order[ ids[len(ids)//2] ] # if outside, use center tactic


    def read_comments( self, comment_objects):
        comments = {}
        for some_cmt_obj in comment_objects:
            components = some_cmt_obj.split( ' ' )
            comments[ components[0] ] = ' '.join( components[1:] )
        return comments


    def parse_vars( self ): # nicer way to store model variables
        if not hasattr( self, 'variables'): return

        res_vars = {}
        some_vars = self.variables.split( ' ' )
        for some_var in some_vars:
            comps = some_var.split('(')
            res_vars[ comps[0] ] = comps[1].split(')')[0]

        self.variables = res_vars


    def files_in_folder( self, folder, filetype=None ):
        paths = []
        for root, _, files in os.walk( folder ):
            for file in files:
                if filetype is not None: # simple check
                    filename, file_extension = os.path.splitext( file )
                    if filetype.lower() not in file_extension.lower(): continue
                paths.append( os.path.join(root, file) )
        return paths


    def set_lims( self ):
        model_mins, model_maxs = [], []
        for ph in self.polyhedrons:
            model_mins.append( self.polyhedrons[ph].lims[0] )
            model_maxs.append( self.polyhedrons[ph].lims[1] )
        
        model_mins = np.vstack(model_mins)
        model_maxs = np.vstack(model_maxs)

        self.lims = [ model_mins.min( axis=0 ), model_maxs.max( axis=0 ) ]
        #print( self.lims )


    def predict( self, x, y, z ):
        # cast to arrays
        if not isinstance( x, np.ndarray ): x=np.array( x )
        if not isinstance( y, np.ndarray ): y=np.array( y )
        if not isinstance( z, np.ndarray ): z=np.array( z )

        preds = np.zeros( shape=x.shape, dtype=np.int32 ) + self.label_outside

        n = len( preds )

        order = self.order[ self.label_outside ]
        for i, (x_, y_, z_) in enumerate( zip(x,y,z) ):
            print( str(int(i/n*100)) + '%', end='\r')

            if not self.p_inside_bbox( x_, y_, z_ ): continue

            for ph in order:
                if self.polyhedrons[ph].p_in_polyhedron( x_, y_, z_ ):
                    preds[i] = ph
                    break
            order = self.order[preds[i]]
        return preds


    def p_inside_bbox( self, x, y, z ): # bbox check for faster prediction
        if (self.lims[0][0]>x) != (self.lims[1][0]>x):
            if (self.lims[0][1]>y) != (self.lims[1][1]>y):
                if (self.lims[0][2]>z) != (self.lims[1][2]>z):
                    return True
        return False


    def even_spred_to_lims( self, idx, num=100 ):

        ax_lims = {
            'Bq': [-0.2, 1.7 ],
            'Fr': [ 0.04, 6  ],
            'Qt': [ 1.5, 50  ],
        }

        ax_name = ['x','y','z'][idx]
        var_name = self.variables[ax_name]
        lims = np.array( ax_lims[var_name] )

        if self.logs[idx]: 
            lims = np.log10( lims )
            return np.logspace( lims[0], lims[1], num=num )
        return np.linspace( lims[0], lims[1], num=num )


    def validate( self, n=100 ):

        # generate meshgrid
        x = self.even_spred_to_lims( 0, num=n )
        y = self.even_spred_to_lims( 1, num=n  )
        z = self.even_spred_to_lims( 2, num=n  )
        xx, yy, zz = np.meshgrid(x, y, z)

        self.xx_flat, self.yy_flat, self.zz_flat = xx.flatten(), yy.flatten(), zz.flatten()

        preds = self.predict( self.xx_flat, self.yy_flat, self.zz_flat )

        k=0
        self.colors = []
        for pred in preds:
            c = ( 1, 1, 1, 0 )
            if pred != self.label_outside: 
                k+=1
                c = tuple( self.polyhedrons[pred].fc + [1] )
            self.colors.append( c )
        print(k)


    def plot( self ):
        plotter = pv.Plotter( notebook=False )
        polyhedron_alpha = None

        if hasattr( self, 'xx_flat' ): # from validate
            if self.logs[0]: self.xx_flat = np.log10(self.xx_flat)
            if self.logs[1]: self.yy_flat = np.log10(self.yy_flat)
            if self.logs[2]: self.zz_flat = np.log10(self.zz_flat)

            points = np.column_stack( (self.xx_flat, self.yy_flat, self.zz_flat) )
            point_cloud = pv.PolyData( points )
            point_cloud['colors'] = self.colors

            #plotter.add_points( point_cloud, scalars='colors', rgba=True )
            plotter.add_mesh(
                point_cloud,
                scalars='colors',
                rgba=True,
                point_size=6.0,
                render_points_as_spheres=True
            )
            polyhedron_alpha = .15
        
        for polyhedron in self.polyhedrons:
            self.polyhedrons[polyhedron].plot( plotter, polyhedron_alpha )

        self.add_coordinate_system( plotter )
        plotter.show( window_size=[1024, 768] )


    def add_coordinate_system( self, plotter):

        ax_lims_ = {}
        for some_var, some_log in zip( ['x','y','z'], self.logs ):
            if some_log:
                ax_lims_[ self.variables[some_var] ] = list( np.log10(self.ax_lims[self.variables[some_var]]) )
            else:
                ax_lims_[ self.variables[some_var] ] = self.ax_lims[self.variables[some_var]]

        ax_lims_comb = list( ax_lims_[self.variables['x']] + ax_lims_[self.variables['y']] + ax_lims_[self.variables['z']] )

        grid_args = dict( 
            axes_ranges = ax_lims_comb,
            bounds = ax_lims_comb,
            font_size = 12,
            xtitle = self.label_dict[self.variables['x']], 
            ytitle = self.label_dict[self.variables['y']], 
            ztitle = self.label_dict[self.variables['z']],
            n_ylabels = 3,
            n_zlabels = 4,
        )

        plotter.show_grid(**grid_args)




    class polyhedron():
        def __init__( self, ply_data, comments, fopacity=1.0, eopacity=.7 ):
            self.faces = []
            self.value = int( comments['value'] )
            self.logs = [ some_scale.lower() == 'log' for some_scale in comments['axis_scales'].split(' ') ]

            vertices = np.array( [list(vertex) for vertex in ply_data['vertex'].data] )
            faces = np.array( [list(face[0]) for face in ply_data['face'].data] )

            self.lims = [ vertices.min( axis=0 ), vertices.max( axis=0 ) ]

            #print( self.value, self.lims )

            # add triangle faces for point_in_polyhedron method
            for face in faces:
                self.faces.append(
                    polyhedron_clf.face(
                        vertices[face[0]],
                        vertices[face[1]],
                        vertices[face[2]],
                        self.logs,
                     )
                )


            # for visualization
            self.fc = [ float(some_color.strip())/255 for some_color in comments['color_rgb'].split(' ') ]
            self.ec = ( 0, 0, 0 )
            self.fopacity = fopacity
            self.eopacity = eopacity

            self.vertices_ = vertices * 1 # copy vertex coordinates

            # apply logs
            for i, some_log in enumerate(self.logs):
                if some_log: self.vertices_[:, i] = np.log10(self.vertices_[:, i])
            self.faces_ = np.hstack([[3, *face] for face in faces])


        def p_in_polyhedron( self, x, y, z ):
            if not self.p_inside_bbox( x,y,z ): return False

            is_inside = False
            for face in self.faces:
                if face.ray_hits_face ( x, y, z ):
                    is_inside = not is_inside
            return is_inside


        def p_inside_bbox( self, x, y, z ): # bbox check for faster prediction
            if (self.lims[0][0]>x) != (self.lims[1][0]>x):
                if (self.lims[0][1]>y) != (self.lims[1][1]>y):
                    if (self.lims[0][2]>z) != (self.lims[1][2]>z):
                        return True
            return False


        def plot( self, plotter, alpha=None ):
            mesh = pv.PolyData( self.vertices_, self.faces_ )

            fopacity=alpha
            eopacity=alpha

            if alpha is None: 
                fopacity=self.fopacity
                eopacity=self.eopacity

            plotter.add_mesh( 
                mesh,
                color=self.fc,
                opacity=fopacity,
                show_edges=True,
                edge_color=self.ec,
                label=str(self.value),
                edge_opacity=eopacity
            )




    class face():
        def __init__( self, p1, p2, p3, logs ):
            self.x = np.array( [p1[0], p2[0], p3[0]] )
            self.y = np.array( [p1[1], p2[1], p3[1]] )
            self.z = np.array( [p1[2], p2[2], p3[2]] )

            self.logx, self.logy, self.logz = logs

            # store triangles sorted by z (used in ray_hits_face)
            idx = self.z.argsort() # ascending order
            self.x = self.x[ idx ]
            self.y = self.y[ idx ]
            self.z = self.z[ idx ]


        def ray_hits_face( self, some_x, some_y, some_z):
            # first a 2D p_in_poly according to Algorithm 1
            is_inside_2D = False
            j=2
            for i in range(3):
                if ( (self.z[i]>some_z) != (self.z[j]>some_z) ):
                    if (some_x < self.x_from_line(some_z, self.x[i], self.z[i], self.x[j], self.z[j], self.logx, self.logz)):
                        is_inside_2D = not is_inside_2D
                j = i
            if not is_inside_2D: return False # short circuit if ray doesn't hit triangle

            # sort point indexes so self.z[ k[0] ] is on opposite side of some_z from other two (true as is_inside_2D==True )
            k = [ 0, 1, 2 ] if (some_z<self.z[1]) else [ 2, 0, 1 ] # differs from Excel sheet implementation: triangles here with ascending z

            # d&e defines the line segment from the intersection face and plane Z=some_z, projected onto the xy-plane
            d_x = self.x_from_line(some_z, self.x[ k[0] ], self.z[ k[0] ], self.x[ k[1] ], self.z[ k[1] ], self.logx, self.logz)
            d_y = self.x_from_line(some_z, self.y[ k[0] ], self.z[ k[0] ], self.y[ k[1] ], self.z[ k[1] ], self.logy, self.logz)
            e_x = self.x_from_line(some_z, self.x[ k[0] ], self.z[ k[0] ], self.x[ k[2] ], self.z[ k[2] ], self.logx, self.logz)
            e_y = self.x_from_line(some_z, self.y[ k[0] ], self.z[ k[0] ], self.y[ k[2] ], self.z[ k[2] ], self.logy, self.logz)

            # define a hit if ray from point ( some_x, some_y, 0 ) in positive y intersects line segment between d & e
            if d_y != e_y:
                return some_y<self.x_from_line(some_x, d_y, d_x, e_y, e_x, self.logy, self.logx)
            else: # handle potential division by zero
                return some_y<d_y


        def x_from_line( self, y, x1, y1, x2, y2, logx, logy): # similar function in "cptu_classification_charts.py"
            if logy and (y>0 and y1>0 and y2>0): # avoids invalid logs
                if logx and x1 > 0 and x2 > 0:  # log-log
                    if y1==y2: return 10**( (np.log10(x1)+np.log10(x2))/2 )
                    return 10**( (np.log10(y / y1) * np.log10(x1 / x2) / np.log10(y1 / y2)) + np.log10(x1) ) # Eq. 8
                else:  # lin-log
                    if y1==y2: return (x1+x2)/2
                    return np.log10(y/y1) * ( (x1-x2) / np.log10(y1/y2) ) + x1 # Eq. 7
            else:
                if logx and (x1>0 and x2>0):  # log-lin
                    if y1==y2: return 10**( (np.log10(x1)+np.log10(x2))/2 )
                    return 10**( ((y-y1) * np.log10(x1/x2) / (y1-y2)) + np.log10(x1) ) # Eq. 6

            # lin-lin ( log(n) where n <= 0 also defaults here )
            if y1==y2: return (x1+x2)/2
            return ( y-y1 ) * ( (x1-x2) / (y1-y2) ) + x1




if __name__=='__main__':
    clf = polyhedron_clf('brittle_screening_2025')
    
    if False: # validate model by classifying - time consuming!
        # example: n=50: 125000 points checked with 24662 (20%) drawn. calculations
        # took 1417s after implementing bbox & ordering optimization (~23.5 mins ).
        clf.validate( n=50 )

    clf.plot()
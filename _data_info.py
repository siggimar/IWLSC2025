from studies import STUDY4 as dataset, get_chart_data, add_data


def stats():
    for soil in dataset:
        for var in soil:
            if isinstance( soil[var] , list ) and len( soil[var] )>0:
                print(var, '\tmin: ', min(soil[var]), '\tmax: ', max(soil[var]))
            else:
                print(var, soil[var])


if __name__=='__main__':
    stats()

def loadPoints(filename):
    input = open(filename, "r")
    
    info = input.readline().split()
    
# number of data points and dimension
    nData = int(info[0]) 
    nDim = int(info[1])
    
# create data matrix
    data = [[0]*nDim for i in range(nData)]

    for i in range(nData):
        info = input.readline().split()
        for j in range(nDim):
            data[i][j] = float(info[j]) 

    return data 

def loadClusters(filename): 
    input = open(filename, "r") 
    
    info = input.readline() 
    
    nData = int(info)
    
    clusters = [0] * nData 
    
    for i in range(nData):
        info = input.readline()
        clusters[i] = int(info)
    
    return clusters


def build_classes_array(data, groundtruth, nb_classes):
    res = [] * nb_classes

    for i in range(nb_classes):
        res.append([])

    for i in range(len(data)):
        res[int(groundtruth[i])].append(data[i])

    return res


def iris_2_2d(point, coord):
    assert len(point) == 4
    assert len(coord) == 2
    return [point[coord[0]], point[coord[1]]]


def liste_class_for_2d(data, coord):
    res = []
    ppc = len(data[0])

    for c in range(len(data)):
        res.append([])

    for c in range(len(data)):
        for i in range(ppc):
            res[c].append(iris_2_2d(data[c][i], coord))

    return res


def liste_point_for_2d(data, coord):
    res = []

    for p in range(len(data)):
        res.append(iris_2_2d(data[p], coord))

    return res


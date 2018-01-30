#!/usr/bin/env python
# -*- coding: utf-8 -*-
# prerequis : librairie pylab installée (regroupe scypy, matplotlib, numpy et ipython)
# codage de la méthode de classification par k plus proches voisins

from pylab import *
from scipy.spatial import distance
from operator import itemgetter
from matplotlib.colors import ListedColormap
from sklearn import neighbors


def affichage_kppv(liste_donnees_classes, point_a_classer, K, voisins, decision):
    assert type(liste_donnees_classes) is list

    nb_classes = len(liste_donnees_classes)
    couleurs = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    formespoints = ['x', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', 'h', 'H', '+', '.', 'D', 'd', ',']
    cmap_light = ListedColormap(['#90CAF9', '#C5E1A5', '#EF9A9A']) # , '#80DEEA', '#CE93D8', '#FFF59D', '#B0BEC5', '#FFFFFF'

    hold(True)
    x_ptnss = []
    y_ptnss = []
    ptnss = []
    c = []
    for classe in range(nb_classes):
        # Affichage des points de référence
        forme = formespoints[classe]
        couleur = couleurs[classe]
        stylepoint = forme + couleur
        for point in liste_donnees_classes[classe]:
            x_ptnss.append(point[0])
            y_ptnss.append(point[1])
            ptnss.append(point)
            c.append(classe)
            plot(point[0], point[1], stylepoint)
            if point in voisins:
                x = [point_a_classer[0], point[0]]
                y = [point_a_classer[1], point[1]]
                plot(x, y, couleur)

    # Affichage du point a classer
    stylepoint = 'D' + ('k' if decision is None else couleurs[decision - 1])
    plot(point_a_classer[0], point_a_classer[1], stylepoint)

    # Calcul et affichage de la frontière de décision
    x_min, x_max = min(x_ptnss) - 1, max(x_ptnss) + 1
    y_min, y_max = min(y_ptnss) - 1, max(y_ptnss) + 1

    h = 0.02  # taille de pas dans le maillage
    xx, yy = meshgrid(arange(x_min, x_max, h), arange(y_min, y_max, h))

    clf = neighbors.KNeighborsClassifier(n_neighbors=K)
    clf.fit(ptnss, c)
    Z = clf.predict(c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    pcolormesh(xx, yy, Z, cmap=cmap_light)

    show()
    hold(False)


def egalite(liste):
    m = max(liste)
    return liste.count(m) > 1


# retourne les K points les plus proches du point à classer, leur distance, leur class et leur coordonnées
def nearest_points(liste_points_classes, point_a_classer, K):
    nb_classes = len(liste_points_classes)
    nb_points_representatifs = len(liste_points_classes[0])

    distances = []  # tableau contenant toutes les distance des points des différentes classes par rapport au point à classer
    for c in range(nb_classes):
        for pt in range(nb_points_representatifs):
            distances.append({'Dist': distance.euclidean(point_a_classer, liste_points_classes[c][pt]), 'Class': c,
                              'Coord': liste_points_classes[c][pt]})

    # recherche des points les plus proches
    sorted_dist = sorted(distances, key=itemgetter('Dist'))

    return sorted_dist[0:K]


# retourne un tableau contenant nombre d’occurences des classes assocíees aux K plus petites distances
def get_class_tab(sorted_dist, nb_classes):
    # Entrée :
    # - sorted_dist : list de dict (Dist, Class, Coord) des K plus proches points

    k = len(sorted_dist)
    res = [0] * nb_classes
    for i in range(k):
        res[int(sorted_dist[i]['Class'])] += 1

    return res


# Cette politique consiste à incrémenter K tant que l'on ne peut pas décider
def heuristique_increase_k(liste_points_classes, point_a_classer, K):
    nb_classes = len(liste_points_classes)
    eg = True
    res = []
    sorted_dist = []
    while eg:
        K += 1
        sorted_dist = nearest_points(liste_points_classes, point_a_classer, K)
        res = get_class_tab(sorted_dist, nb_classes)
        eg = egalite(res)

    return (argmax(res) + 1), K, sorted_dist


# Cette politique consiste à ne pas décider
def heuristique_no_decision(liste_points_classes, point_a_classer, K):
    sorted_dist = nearest_points(liste_points_classes, point_a_classer, K)
    return None, K, sorted_dist


def decision_kppv(liste_points_classes, point_a_classer, K, politique):
    assert type(liste_points_classes) is list
    # decision_kppv
    #
    # Entrée :
    # - liste_points_classes : liste pour chaque classe de liste de
    #   points représentatifs de la classe
    # - point_a_classer :
    # - K : entier, nombre de voisins a prendre en compte
    # Sortie :
    # - numero de la classe la plus proche (0 si pas de décision)

    nb_classes = len(liste_points_classes)

    sorted_dist = nearest_points(liste_points_classes, point_a_classer, K)

    # décision sur les classes
    res = get_class_tab(sorted_dist, nb_classes)

    # Si on a égalité on prend un décision en fonction de la politique choisie
    new_k = K
    if egalite(res):
        classe_la_plus_proche, new_k, sorted_dist = politique(liste_points_classes, point_a_classer, K)
    else:
        classe_la_plus_proche = argmax(res) + 1

    points = []
    for i in range(len(sorted_dist)):
        points.append(sorted_dist[i]['Coord'])

    return classe_la_plus_proche, new_k, points


# Programme principal : préparation points représentatifs et
#                       test sur 3 points


donnees_classe1 = [[0.94922183000000004, 0.30706191999999999], [0.13517493999999999, 0.51524634000000002],
                   [0.26140632000000003, -0.94148577], [-0.16233766999999999, -0.14605462999999999],
                   [-0.53201138000000003, 1.6821036], [-0.87572934999999996, -0.48381505000000002],
                   [-0.71200454999999996, -1.1742123], [-0.19223952, -0.27407023000000003],
                   [1.5300724999999999, -0.24902473999999999], [-1.0642134000000001, 1.6034573000000001],
                   [1.2346790999999999, -0.22962645000000001], [-1.5061597, -0.44462782000000001],
                   [-0.15594104, 0.27606825000000002], [-0.26116365000000002, 0.44342190999999997],
                   [0.39189421000000002, -1.2506789], [-0.94796091999999998, -0.74110609000000005],
                   [-0.50781754999999995, -0.32057551000000001], [0.012469041, -3.0291773000000002],
                   [-0.45701464000000003, 1.2424484]]
donnees_classe2 = [[0.74115313999999999, 3.8621732999999998], [3.3187652000000001, 1.6923117000000001],
                   [2.566408, 3.3426244999999999], [6.5783969000000004, 5.7694369999999999],
                   [1.6501131, 6.0349234999999997], [3.7254041999999998, 2.9369451], [3.7147429000000001, 2.7950339],
                   [2.8758556999999998, 4.4896976000000004], [4.4090344999999997, 4.4171924000000002],
                   [3.6714970999999998, 1.7925131000000001], [3.7172386999999998, 4.6302352999999998],
                   [3.4888938, 4.0346929999999999], [3.7268851000000001, 2.6965591],
                   [3.2938714999999998, 2.2127172000000002], [3.8883956, 1.8529298999999999],
                   [1.9311294999999999, 2.1905013000000002], [0.055715837999999997, 4.4383803000000004],
                   [3.3251905000000002, 2.2450717], [4.3702984999999996, 1.2884836]]
donnees_classe3 = [[1.5168790000000001, -1.0202644000000001], [2.5530050000000002, 0.10965859],
                   [4.1287364999999996, -0.28996304000000001], [4.2615506999999999, 0.47542480999999998],
                   [4.1741168000000002, 0.12694707], [2.3431841000000002, -1.4813991], [3.1554890000000002, 0.81855137],
                   [2.7074118999999999, -0.54078641999999999], [2.6913581999999998, -1.0965933005499999],
                   [2.9201071999999999, 0.89847599], [3.1837034000000002, 0.29079012999999998],
                   [3.1129446999999999, 0.43995219000000002], [3.1016623999999999, 2.7873351999999998],
                   [1.8333349999999999, -1.8542991], [1.8593189000000001, -1.0933435],
                   [2.5663906999999999, -0.16846987999999999], [2.7814663999999998, 0.54133443999999997],
                   [3.3892661999999998, 0.75122898000000005], [4.7782558999999996, 1.2230626]]
donnee_test_classe1 = [0.64867925999999998, 0.82572714999999997]
donnee_test_classe2 = [2.8023018, 1.7921545000000001]
donnee_test_classe3 = [3.6829643000000001, 0.9706087000000001]
donnee_test = [1.9, 1.02572714999999997]  # illustration figure sujet


'''
donnees_classe1 = [[3.6484028273127254, 0.073429844289042881], [2.330634259457685, 0.21748520551489497], [2.644948143301538, 0.20411304520644272], [2.7753121336336251, 0.18681685779827709], [2.1983916903402347, 0.19693911318338075], [2.836900928011457, 0.16884540288894506], [2.6235769902396133, 0.14047193438597244], [2.7157154911382348, 0.18092519035140286], [2.6662657025141492, 0.17384029751490698], [2.7900235283901487, 0.17124467162767718], [2.3025641883523864, 0.19080233853944761], [2.50402381354598, 0.26502653237173324], [2.4698500051963022, 0.23443828139261397], [2.5051902610502612, 0.23067614484914289], [2.4684409160318621, 0.21863502590393655], [3.0295456522668971, 0.086190046340667389], [2.4470385350120138, 0.21074952985853601], [2.5841314160351514, 0.1399509161051439], [2.9340821512886901, 0.13392607949486274]]
donnees_classe2 = [[4.0958394694071085, 0.62090992691837554], [3.0515154047252695, 0.53512517722257091], [2.9491682603712408, 0.44028070952077253], [3.2829314223632156, 0.49129132399409814], [3.2997815280896532, 0.46202148420611983], [3.1994813507553297, 0.51730303394481048], [3.2739397765597413, 0.42456810719843918], [3.0094501931685627, 0.53307647477381548], [3.1121716887339406, 0.61264659221630813], [3.1567904507066951, 0.56034384641028545], [3.1666032722891706, 0.47572772737933283], [2.8978523007369641, 0.39472962675440465], [3.3719622811831127, 0.57256856612568585], [3.0025688112273086, 0.5581421242147987], [3.2243758188595679, 0.52064997498366095], [3.0110164875805565, 0.43111214123321362], [3.1485374067025722, 0.64330361652352819], [3.2302342511064359, 0.62838600861174609], [3.4725953755453878, 0.43200027260585433]]
donnees_classe3 = [[3.836285556162859, 0.17488885815676444], [3.9046036496933407, 0.21995742377870348], [4.1545108753116367, 0.19645400534459045], [4.1987833316122352, 0.19703304602057539], [3.8715397281540196, 0.18336748445715684], [4.1631952751744903, 0.21583751780035143], [3.8067755719682346, 0.13029984504649231], [4.2331995779590086, 0.16583095278355137], [4.0664834258300173, 0.26180733777608278], [3.8539747179857335, 0.17215241559346361], [4.1093448646782491, 0.17765106796797839], [3.7176133236929481, 0.24048737653006386], [4.3310642505061496, 0.18847876748651521], [3.5832847883882, 0.22498735527596694], [3.999611132714854, 0.12367136651967732], [4.2664222137560586, 0.28957677201563969], [3.9681326235446828, 0.21086457727627866], [3.7197425382728775, 0.24458119801735781], [3.6350175478315383, 0.23427148092916744]]


donnee_test_classe1 = [3.0299364237682695, 0.069933489299203899]
donnee_test_classe2 = [3.3156657804150163, 0.67327144428452557]
donnee_test_classe3 = [4.0484484569058861, 0.24216712711692681]
'''

liste_donnees_classes = [donnees_classe1, donnees_classe2, donnees_classe3]



if __name__ == '__main__':
    K = 3
    politique = heuristique_increase_k

    print("Decision par K-PPV, avec K = %d" % K)
    decision1, K, voisins = decision_kppv(liste_donnees_classes, donnee_test_classe1, K, politique)
    print("La donnee de classe 1 a ete reconnue comme une donnee de classe %s" % decision1)
    figure()
    affichage_kppv(liste_donnees_classes, donnee_test_classe1, K, voisins, decision1)

    decision2, K, voisins = decision_kppv(liste_donnees_classes, donnee_test_classe2, K, politique)
    print("La donnee de classe 2 a ete reconnue comme une donnee de classe %s" % decision2)
    figure()
    affichage_kppv(liste_donnees_classes, donnee_test_classe2, K, voisins, decision2)

    decision3, K, voisins = decision_kppv(liste_donnees_classes, donnee_test_classe3, K, politique)
    print("La donnee de classe 3 a ete reconnue comme une donnee de classe %s" % decision3)
    figure()
    affichage_kppv(liste_donnees_classes, donnee_test_classe3, K, voisins, decision3)

    print("Cas d'indécision (K=5)")
    donnee_test_indecidable = [1.65, 1.02]
    K = 5
    decision, K, voisins = decision_kppv(liste_donnees_classes, donnee_test_indecidable, K, politique)
    print("La donnee a ete reconnue comme une donnee de classe %s Normalement : indecidable." % decision)
    affichage_kppv(liste_donnees_classes, donnee_test_indecidable, K, voisins, decision)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# prerequis : librairie pylab installée (regroupe scypy, matplotlib, numpy et ipython)
# codage de la méthode de classification par k plus proches voisins

from pylab import *
from scipy.spatial import distance
from operator import itemgetter
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from LoadData import *
from utils import *


def affichage_kppv(liste_donnees_classes, point_a_classer, K, voisins, decision, display_border=False,
                   display_lines=True):
    assert type(liste_donnees_classes) is list

    nb_classes = len(liste_donnees_classes)
    couleurs = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    formespoints = ['x', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', 'h', 'H', '+', '.', 'D', 'd', ',']
    cmap_light = ListedColormap(
        ['#90CAF9', '#C5E1A5', '#EF9A9A'])  # , '#80DEEA', '#CE93D8', '#FFF59D', '#B0BEC5', '#FFFFFF'

    figure(facecolor="white")

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
            if display_lines:
                if point in voisins:
                    x = [point_a_classer[0], point[0]]
                    y = [point_a_classer[1], point[1]]
                    plot(x, y, couleur)

    # Affichage du point a classer
    stylepoint = 'D' + ('k' if decision is None else couleurs[decision - 1])
    plot(point_a_classer[0], point_a_classer[1], stylepoint)

    # Calcul et affichage de la frontière de décision
    if display_border:
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


def affichage_kppv_4d(liste_donnees_classes, donnee_test_classe1, K, voisins, decision1, display_border=False,
                      display_lines=False):
    coord = [2, 3]
    affichage_kppv(liste_class_for_2d(liste_donnees_classes, coord), iris_2_2d(donnee_test_classe1, coord), K,
                   liste_point_for_2d(voisins, coord), decision1, display_border=display_border,
                   display_lines=display_lines)
    coord = [0, 1]
    affichage_kppv(liste_class_for_2d(liste_donnees_classes, coord), iris_2_2d(donnee_test_classe1, coord), K,
                   liste_point_for_2d(voisins, coord), decision1, display_border=display_border,
                   display_lines=display_lines)


def affichage_kppv_list_to_test(liste_donnees_classes, liste_point_a_classer, liste_decision):
    assert type(liste_donnees_classes) is list
    assert type(liste_point_a_classer) is list
    assert type(liste_decision) is list
    assert len(liste_decision) == len(liste_point_a_classer)

    nb_classes = len(liste_donnees_classes)
    couleurs = ['b', 'g', 'c', 'y', 'r', 'm', 'k', 'w']
    formespoints = ['x', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', 'h', 'H', '+', '.', 'D', 'd', ',']

    figure(facecolor="white")

    hold(True)
    for classe in range(nb_classes):
        # Affichage des points de référence
        forme = formespoints[classe]
        couleur = couleurs[classe + 2]
        stylepoint = forme + couleur
        for point in liste_donnees_classes[classe]:
            plot(point[0], point[1], stylepoint, markersize=3)

    # Affichage des points a classer
    for i in range(len(liste_point_a_classer)):
        forme = formespoints[(liste_decision[i] - 1) % len(formespoints)]
        couleur = 'k' if liste_decision[i] is None else couleurs[(liste_decision[i] - 1) % len(couleurs)]
        stylepoint = forme + couleur
        plot(liste_point_a_classer[i][0], liste_point_a_classer[i][1], stylepoint)

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


def heuristique_mean(liste_points_classes, point_a_classer, K):
    nb_classes = len(liste_points_classes)
    sorted_dist = nearest_points(liste_points_classes, point_a_classer, K)
    res = get_class_tab(sorted_dist, nb_classes)
    m = max(res)

    equ_classes = []
    for c in range(nb_classes):
        if res[c] == m:
            equ_classes.append(c)

    nb_class_equ = len(equ_classes)

    mean_by_class = [float("inf")] * nb_classes
    for c in range(nb_class_equ):
        mean_c = 0
        for i in range(len(sorted_dist)):
            if sorted_dist[i]['Class'] == equ_classes[c]:
                mean_c += sorted_dist[i]['Dist']
        mean_c = mean_c / res[equ_classes[c]]
        mean_by_class[equ_classes[c]] = mean_c

    best_class = argmin(mean_by_class)

    return (best_class + 1), K, sorted_dist


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



if __name__ == '__main__':
    dataFilename = 'data/iris.data'
    groundtruthFilename = 'data/iris.ground'

    data_iris = loadPoints(dataFilename)
    groundtruth_iris = loadClusters(groundtruthFilename)

    data_to_test = loadPoints('data/self_test2.data')

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
                       [1.6501131, 6.0349234999999997], [3.7254041999999998, 2.9369451],
                       [3.7147429000000001, 2.7950339],
                       [2.8758556999999998, 4.4896976000000004], [4.4090344999999997, 4.4171924000000002],
                       [3.6714970999999998, 1.7925131000000001], [3.7172386999999998, 4.6302352999999998],
                       [3.4888938, 4.0346929999999999], [3.7268851000000001, 2.6965591],
                       [3.2938714999999998, 2.2127172000000002], [3.8883956, 1.8529298999999999],
                       [1.9311294999999999, 2.1905013000000002], [0.055715837999999997, 4.4383803000000004],
                       [3.3251905000000002, 2.2450717], [4.3702984999999996, 1.2884836]]
    donnees_classe3 = [[1.5168790000000001, -1.0202644000000001], [2.5530050000000002, 0.10965859],
                       [4.1287364999999996, -0.28996304000000001], [4.2615506999999999, 0.47542480999999998],
                       [4.1741168000000002, 0.12694707], [2.3431841000000002, -1.4813991],
                       [3.1554890000000002, 0.81855137],
                       [2.7074118999999999, -0.54078641999999999], [2.6913581999999998, -1.0965933005499999],
                       [2.9201071999999999, 0.89847599], [3.1837034000000002, 0.29079012999999998],
                       [3.1129446999999999, 0.43995219000000002], [3.1016623999999999, 2.7873351999999998],
                       [1.8333349999999999, -1.8542991], [1.8593189000000001, -1.0933435],
                       [2.5663906999999999, -0.16846987999999999], [2.7814663999999998, 0.54133443999999997],
                       [3.3892661999999998, 0.75122898000000005], [4.7782558999999996, 1.2230626]]
    donnee_test_classe1 = [0.64867925999999998, 0.82572714999999997]
    donnee_test_classe2 = [2.8023018, 1.7921545000000001]
    donnee_test_classe3 = [3.6829643000000001, 0.9706087000000001]

    liste_donnees_classes = [donnees_classe1, donnees_classe2, donnees_classe3]

    K = 3
    politique = heuristique_increase_k
    display_border = True
    display_lines = True

    print("Decision par K-PPV, avec K = %d" % K)
    decision1, K, voisins = decision_kppv(liste_donnees_classes, donnee_test_classe1, K, politique)
    print("La donnee de classe 1 a ete reconnue comme une donnee de classe %s" % decision1)
    affichage_kppv(liste_donnees_classes, donnee_test_classe1, K, voisins, decision1, display_border=display_border,
                   display_lines=display_lines)

    decision2, K, voisins = decision_kppv(liste_donnees_classes, donnee_test_classe2, K, politique)
    print("La donnee de classe 2 a ete reconnue comme une donnee de classe %s" % decision2)
    affichage_kppv(liste_donnees_classes, donnee_test_classe2, K, voisins, decision2, display_border=display_border,
                   display_lines=display_lines)

    decision3, K, voisins = decision_kppv(liste_donnees_classes, donnee_test_classe3, K, politique)
    print("La donnee de classe 3 a ete reconnue comme une donnee de classe %s" % decision3)
    affichage_kppv(liste_donnees_classes, donnee_test_classe3, K, voisins, decision3, display_border=display_border,
                   display_lines=display_lines)

    print("Cas d'indécision (K=5)")
    donnee_test_indecidable = [1.65, 1.02]
    K = 5
    decision, K, voisins = decision_kppv(liste_donnees_classes, donnee_test_indecidable, K, politique)
    print("La donnee a ete reconnue comme une donnee de classe %s Normalement : indecidable." % decision)
    affichage_kppv(liste_donnees_classes, donnee_test_indecidable, K, voisins, decision, display_border=display_border,
                   display_lines=display_lines)

    print("Test sur les données IRIS")

    donnee_test_iris_c1 = [4.7, 3.5, 2., 0.5]
    donnee_test_iris_c2 = [5.2, 2.5, 4.2, 1.4]
    donnee_test_iris_c3 = [7.5, 2.7, 6., 2.]
    nb_classes = 3
    display_border = False
    display_lines = True
    liste_donnees_classes = build_classes_array(data_iris, groundtruth_iris, nb_classes)

    decision1, K, voisins = decision_kppv(liste_donnees_classes, donnee_test_iris_c1, K, politique)
    print("La donnee de classe 1 a ete reconnue comme une donnee de classe %s" % decision1)
    affichage_kppv_4d(liste_donnees_classes, donnee_test_iris_c1, K, voisins, decision1, display_border=display_border,
                      display_lines=display_lines)

    decision2, K, voisins = decision_kppv(liste_donnees_classes, donnee_test_iris_c2, K, politique)
    print("La donnee de classe 2 a ete reconnue comme une donnee de classe %s" % decision2)
    affichage_kppv_4d(liste_donnees_classes, donnee_test_iris_c2, K, voisins, decision2, display_border=display_border,
                      display_lines=display_lines)

    decision3, K, voisins = decision_kppv(liste_donnees_classes, donnee_test_iris_c3, K, politique)
    print("La donnee de classe 3 a ete reconnue comme une donnee de classe %s" % decision3)
    affichage_kppv_4d(liste_donnees_classes, donnee_test_iris_c3, K, voisins, decision3, display_border=display_border,
                      display_lines=display_lines)

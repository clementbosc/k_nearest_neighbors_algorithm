# coding=utf-8


def min_max_liste(data, dim):
    m, M = data[0][dim], data[1][dim]
    for i in range(len(data)):
        el = data[i][dim]
        if el < m:
            m = el
        if el > M:
            M = el
    return m, M


def affichage_donnees_old(data, centres, clusterID):
    import matplotlib.pyplot as plt
    assert type(data) is list
    assert type(centres) is list
    plt.cla()
    couleurs = [ 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w' ]
    styles = [ 'x', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', '.', 'D', 'd' ]
    formes = []
    formes_ctr = []

    # pour fixer les intervalles des axes:
    min_x, max_x = min_max_liste(data, dim=0)
    min_y, max_y = min_max_liste(data, dim=1)

    nb = len(centres)
    no_couleur = 0
    style = styles.pop(0)
    for groupe in range(nb):
        if no_couleur > len(couleurs):
            style = styles.pop(0)
            no_couleur = 0
        formes.append(style + couleurs[no_couleur])
        formes_ctr.append("o" + couleurs[no_couleur])
        no_couleur += 1

    plt.hold(True)
    for groupe in range(nb):
        forme = formes[groupe]
        plt.plot(centres[groupe][0], centres[groupe][1], formes_ctr[groupe])

    for ind in range(len(data)):
        point = data[ind]
        pointID = clusterID[ind]
        plt.plot(point[0],point[1],formes[pointID])
    plt.hold(False)
    plt.axis([min_x, max_x, min_y, max_y])
    # plt.axis([2, 5, -1, 2])
    plt.show()
    # plt.savefig('kmeans_4nuages.png')


def affichage_donnees(data, cluster_id):
    import matplotlib.pyplot as plt
    plt.cla()
    formes = ['xg', 'or', 'vb']

    # # pour fixer les intervalles des axes:
    # min_x, max_x = min_max_liste(data, dim=0)
    # min_y, max_y = min_max_liste(data, dim=1)

    x_ind, y_ind = 2, 3

    # print formes
    for ind in range(len(data)):
        point = data[ind]
        point_id = cluster_id[ind]
        # print pointID, formes[pointID], x_ind, y_ind
        plt.plot(point[x_ind], point[y_ind], formes[point_id])
        # plt.plot(point[x_ind],point[y_ind], 'ob')
    # plt.hold(False)

    #for groupe in range(nb):
    #    # plt.plot(centres[groupe][x_ind], centres[groupe][y_ind], formes_ctr[groupe], markersize=14)
    #    plt.plot(centres[groupe][x_ind], centres[groupe][y_ind], 'hk', markersize=14)
    #    plt.plot(centresInitiaux[groupe][x_ind], centresInitiaux[groupe][y_ind], 'hc', markersize=14)

    plt.axis([0, 8, 0, 3])

    plt.show()

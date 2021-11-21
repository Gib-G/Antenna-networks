###0) Modules et constantes


import numpy as np
import math as ma
import matplotlib.pyplot as plt
import matplotlib
import mpl_toolkits.mplot3d as mpl

# Constantes:

m=9.1e-31 # Masse d'un électron.
qe=1.6e-19 # Charge élémentaire.
c=3e8 # Célérité de la lumière dans le vide.
epsilon0=8.854e-12 # Permittivité diélectrique du vide.
mu0=1.256e-6 # Perméabilité magnétique du vide.
p0=1000

# Variables du problème:

w=2*ma.pi*(100000) # Pulsation du moment dipôlaire.
theta= ma.pi # Deuxième coordonnée sphérique (0<=theta<=ma.pi).
r=10 # Distance à la source en mètres.
l_onde=2 # Longueur d'onde (valeur arbitraire).


###1) Propagation ionosphérique
    

def moyenne(fonction,a,b,n):
# Renvoie la valeur moyenne d'une fonction sur un intervalle [a,b] (avec a<b)
# en utilisant la méthode d'intégration des trapèzes. L'intervalle [a,b] est
# subdivisés en 'n' intervalles. 
    S=0  
    h=(b-a)/n
    for k in range(n):
        S=S+fonction(a+k*h)+fonction(a+(k+1)*h)
    return (S/(2*n))
    
    
def discretisation_indice_moy(profil_indice,altitude_début,altitude_fin,nb_strates):
# Pour une fonction 'profil_indice' donnée, renvoie la liste formée par la
# concaténation
# [profil_indice(altitude_début)]+L+[profil_indice(altitude_fin)] où L est une
# liste de 'nb_strates' valeurs contenant les valeurs moyennes de
# 'profil_indice' sur chaque strate du modèle ionosphérique retenu. 
    L=[profil_indice(altitude_début)]
    pas = (abs(altitude_fin - altitude_début))/nb_strates
    for k in range(nb_strates):
        L.append(moyenne(profil_indice,altitude_début + k*pas,altitude_début + (k+1)*pas,1000))
    L.append(profil_indice(altitude_fin))
    return L
    
    
# On utilise le profil de densité d'électrons au sein de l'atmosphère suivant.
# Ce dernier est une liste de listes de la forme
# [[altitude en km,densité d'électrons en cm-3],...] :

profil_electrons_cm3=[[0,0],[5,1600],[7,2800],[12,4400],[17,6500],[21,1.020e4],[24,1.550e4],[31,2.410e4],[38,3.690e4],[48,6.420e4],[55,9.810e4],[67,1.682e5],[74,2.613e5],[76,3.284e5],[83,4.334e5],[93,5.628e5],[98,6.958e5],[105,8.888e5],[117,1.099e6],[126,1.337e6],[136,1.626e6],[160,2.328e6],[176,2.786e6],[190,3.227e6],[207,3.559e6],[224,3.862e6],[252,4.122e6],[281,3.990e6],[300,3.738e6],[319,3.389e6],[343,2.974e6],[367,2.526e6],[381,2.328e6],[398,2.043e6],[410,1.823e6],[426,1.626e6],[440,1.450e6],[460,1.294e6],[474,1.173e6],[483,1.046e6],[505,8.888e5],[524,7.928e5],[540,6.958e5],[560,6.207e5],[579,5.537e5],[600,4.939e5],[619,4.478e5],[643,4.127e5],[660,3.742e5],[679,3.449e5],[698,3.338e5],[714,3.127e5],[736,2.929e5],[755,2.789e5],[776,2.656e5],[810,2.529e5],[840,2.408e5],[871,2.331e5],[890,2.256e5],[912,2.220e5],[933,2.148e5],[957,2.148e5],[998,2.114e5]]	

profil_electrons_m3=[[k[0],k[1]*(10**6)] for k in profil_electrons_cm3] # Les
# densités d'électrons sont exprimées en m-3.


# Allure du profil de densité d'électrons dans l'atmosphère en fonction de
# l'altitude :


liste_alt=[k[0] for k in profil_electrons_m3]
liste_den=[k[1] for k in profil_electrons_m3]
plt.figure()
plt.plot(liste_alt,liste_den)
plt.title('Densité d\'électrons dans l\'atmosphère en fonction de l\'altitude')
plt.xlabel('Altitude (en km)')
plt.ylabel('Densité d\'électrons (en électrons par m^3)')
plt.show()


# Le profil de densité d'électrons discret est rendu continu sur l'intervalle
# d'altitudes étudié grâce à la fonction 'profil_emp_electrons'. Celle-ci
# prend en entrée un profil discret de densité d'électrons sur un intervalle
# d'altitudes donné ('profil') et renvoie la valeur de la densité d'électrons
# à l'altitude 'alt' comprise dans l'intervalle d'altitudes sur lequel le
# profil de densité d'électrons est défini.


def profil_emp_electrons(profil,alt):
    if alt<0:
        return 0
    elif alt>=profil[len(profil) - 1][0]:
        return profil[len(profil) - 1][1]
    else:
        k=0
        while k<len(profil) and alt>=profil[k][0]:
            k=k+1
        d1=abs(alt - profil[k-1][0])
        d2=abs(alt - profil[k][0])
        D=abs(profil[k-1][0] - profil[k][0])
        return (d2*profil[k-1][1] + d1*profil[k][1])/D
        
        
def profil_emp_indice(profil,freq,alt):
# Renvoie la valeur de l'indice de réfraction de l'air ionisé pour une onde de
# fréquence 'freq' à l'altitude 'alt' pour un profil continu de densité
# d'électrons dans l'atmosphère ('profil') en fonction de l'altitude.
    return ma.sqrt(1 - (profil_emp_electrons(profil,alt)*qe**2)/(4*(ma.pi**2)*m*epsilon0*freq**2))
    
    
def profil_emp_indice2(alt):
# Renvoie le même résultat que la fonction précédente spécifiquement pour le
# profil de densité d'électrons étudié ('profil_electrons_m3') à la fréquence
# de 50 MHz.
    return profil_emp_indice(profil_electrons_m3,5e7,alt)
    

# Allure de l'indice de réfraction de l'air ionisé en fonction de l'altitude
# pour le profil de densité d'électrons étudié ('profil_electrons_m3') et une
# onde de fréquence 50 MHz :


h=0.001
liste_alt=[k*h for k in range(1000000)]
liste_n=[profil_emp_indice(profil_electrons_m3,5e7,alt) for alt in liste_alt]
plt.figure()
plt.plot(liste_alt,liste_n)
plt.title('Indice de réfraction de l\'air en fonction de l\'altitude')
plt.xlabel('Altitude (en km)')
plt.ylabel('Indice de réfraction de l\'air')
plt.show()


# On détermine le "trajet" de l'onde électromagnétique dans les conditions de
# l'optique géométrique grâce à la fonction 'trajet'. Celle-ci prend en
# argument un angle d'incidence initial de l'onde au niveau de son point
# d'émission avant sa pénétration dans l'atmosphère, un profil continu de
# l'indice de réfraction de l'air ionisé en fonction de l'altitude
# ('profil_indice'), une altitude initiale et finale de la couche
# atmosphérique considérée subdivisée en 'nb_strates' strates d'indices de
# réfraction propres.
    

def trajet(incidence,profil_indice,altitude_début,altitude_fin,nb_strates):
# Renvoie un tuple dont le premier élément est un tuple contenant le trajet de
# l'onde sous forme de deux listes, l'une dénotant les déplacements
# horizontaux de l'onde ('ListeX') et l'autre ses déplacements verticaux
# ('ListeY'). Le deuxième élément,'k', correspond à l'indice de la strate où a
# lieu la réflexion
# totale. Si il n'y a pas de réflexion totale, k est supérieur au nombre de
# strates choisi.
    pas=(abs(altitude_fin - altitude_début))/nb_strates
    L=discretisation_indice_moy(profil_indice,altitude_début,altitude_fin,nb_strates)
    ListeX=[0]
    ListeY=[altitude_début]
    k=0
    while k<len(L) and abs((L[0]/L[k])*ma.sin(incidence))<1: # La 2e condition
    # correspond à une situation de réflexion totale.
        i=ma.asin((L[0]/L[k])*ma.sin(incidence))
        x,y=len(ListeX)-1,len(ListeY)-1
        ListeX.append(ListeX[x] + pas*ma.tan(i))
        ListeY.append(ListeY[y] + pas)
        k=k+1
    if k<len(L): # Si il y a réflexion totale avant que l'onde n'ait traversé
    # toutes les strates, le rayon est renvoyé vers la Terre.
        n=L[k]
        incidence2=-ma.asin((L[0]/L[k-1])*ma.sin(incidence))
        x,y=len(ListeX)-1,len(ListeY)-1
        ListeX.append(ListeX[x] - pas*ma.tan(incidence2))
        ListeY.append(ListeY[y] - pas)
        for j in range(k-1,-1,-1):
            i=ma.asin((n/L[j])*ma.sin(incidence2))
            x,y=len(ListeX)-1,len(ListeY)-1
            ListeX.append(ListeX[x] - pas*ma.tan(i))
            ListeY.append(ListeY[y] - pas)
    return((ListeX,ListeY),k)
    
    
# Tracé des trajets d'une onde de fréquence 50 MHz en fonction de l'angle
# d'incidence pour le profil d'indices de réfraction étudié :
  

ListeX,ListeY=trajet(1.38,profil_emp_indice2,0,1000,400)[0]
plt.figure()
plt.plot(ListeX,ListeY)
plt.title("Trajet d'un rayon d'incidence 79°, 400 strates, 50 MHz")
plt.xlabel("Distance au point d'émission (en km)")
plt.ylabel("Altitude (en km)")
plt.show()

ListeX,ListeY=trajet(1.2,profil_emp_indice2,0,1000,400)[0]
plt.figure()
plt.plot(ListeX,ListeY)
plt.title("Trajet d'un rayon d'incidence 69°, 400 strates, 50 MHz")
plt.xlabel("Distance au point d'émission (en km)")
plt.ylabel("Altitude (en km)")
plt.show()
    
ListeX,ListeY=trajet(1.17,profil_emp_indice2,0,1000,400)[0]
plt.figure()
plt.plot(ListeX,ListeY)
plt.title("Trajet d'un rayon d'incidence 67°, 400 strates, 50 MHz")
plt.xlabel("Distance au point d'émission (en km)")
plt.ylabel("Altitude (en km)")
plt.show()


# On étudie les variations de l'altitude de réflexion totale de l'onde en
# fonction de son angle d'incidence :


def altitude(angle_début,angle_fin,nb_points,profil_indice):
    ListeX=np.linspace(angle_début,angle_fin,nb_points)
    ListeY=[(6/100)*trajet(x,profil_indice,0,6,100)[1] for x in ListeX]
    plt.figure()
    plt.plot(ListeX,ListeY)
    plt.show()
    
    
## Essais avec des profils de densité d'électrons différents


def interpolation_lagrange(liste_points,x):
# Renvoie la valeur du polynôme de Lagrange associé aux points de
# 'liste_points' évalué en 'x'. 'liste_points' est une liste de tuples
# représentant les coordonnées des points d'interpolation sous la forme
# (abscisse,ordonnée).
    S=0
    for i in range(len(liste_points)):
        P=1
        xi,yi=liste_points[i]
        for j in range(len(liste_points)):
            if j!=i:
                P=P*(x-liste_points[j][0])/(xi-liste_points[j][0])
        S=S+P*yi
    return(S)

    
def interpolation_lagrange2(x):
# Renvoie le même résultat que la fonction précédente pour les points de
# coordonnées (0,1), (500,0.5) et (1000,1).
    return(interpolation_lagrange([(0,1),(500,0.5),(1000,1)],x))


# Profil d'indices de réfraction considéré (parabolique) :


ListeT=np.linspace(0,1000,10000)
ListeY=[interpolation_lagrange2(x) for x in ListeT]
plt.figure()
plt.plot(ListeT,ListeY)
plt.title("Indice de réfraction de l'air ionisé en fonction de l'altitude (approximation parabolique)")
plt.xlabel("altitude (en km)")
plt.ylabel("indice de réfraction")
plt.show()


# Trajets d'une onde de fréquence 50 MHz en fonction de l'angle d'incidence
# pour le profil d'indices de réfraction parabolique :


ListeX,ListeY=trajet(0.49,interpolation_lagrange2,0,1000,400)[0]
plt.figure()
plt.plot(ListeX,ListeY)
plt.title("Trajet d'un rayon d'incidence 28°, 400 strates, 50 MHz")
plt.xlabel("distance au point d'émission (en centaine de km)")
plt.ylabel("Altitude (en centaines de km)")
plt.show()

ListeX,ListeY=trajet(0.53,interpolation_lagrange2,0,1000,400)[0]
plt.figure()
plt.plot(ListeX,ListeY)
plt.title("Trajet d'un rayon d'incidence 30°, 400 strates, 50 MHz")
plt.xlabel("distance au point d'émission (en centaine de km)")
plt.ylabel("Altitude (en centaines de km)")
plt.show()


## Coefficients de réflexion et transmission, approche ondulatoire

    
def reflexion(n1,n2,i):
# Renvoie la valeur du coefficient de réflexion en incidence oblique entre les
# milieux 1 et 2. 'n1' et 'n2' sont les indices de réfraction des milieux 1 et
# 2 respectivement. 'i' est l'angle d'incidence de l'onde.
    a=ma.sqrt(1 - ((n1/n2)*ma.sin(i))**2)
    return (n1*ma.cos(i) - n2*a)/(n1*ma.cos(i) + n2*a)
    
    
def ref_tot(n1,n2,i1,i2):
# Renvoie la valeur du coefficient de réflexion en incidence oblique entre les
# milieux 1 et 2 en cas de réflexion totale. 'n1' et 'n2' sont les indices de
# réfraction des milieux 1 et 2 respectivement. 'i1' est l'angle d'incidence
# initial de l'onde et 'i2', l'angle d'incidence du rayon réfracté.
    return (n1*ma.cos(i1) - n2*ma.cos(i2))/(n1*ma.cos(i1) + n2*ma.cos(i2))
    
    
def transmission(n1,n2,i):
# Renvoie la valeur du coefficient de transmission en incidence oblique entre
# les milieux 1 et 2. 'n1' et 'n2' sont les indices de réfraction des milieux
# 1 et 2 respectivement. 'i' est l'angle d'incidence de l'onde.
    a=ma.sqrt(1 - ((n1/n2)*ma.sin(i))**2)
    return (2*n1*ma.cos(i))/(n1*ma.cos(i) + n2*a)
    
    
def trajet2(incidence,profil_indice,altitude_début,altitude_fin,nb_strates):
# Même spécification que la fonction 'trajet' en prenant en compte
# coefficients de réflexion et transmission.
    pas=(abs(altitude_fin - altitude_début))/nb_strates
    L=Discretisation_Indice_moy2(profil_indice,altitude_début,altitude_fin,nb_strates)
    ListeX=[0]
    ListeY=[altitude_début]
    k=0
    n1=L[0]
    dist,prop=[0],[1]
    t,r=transmission(L[0],L[1],i),reflexion(L[0],L[1],i)
    while k<(len(L)-1) and abs((n1/L[k+1])*ma.sin(incidence))<1 and r<=t:
        i=ma.asin((n1/L[k+1])*ma.sin(incidence))
        x,y,d,p=len(ListeX)-1,len(ListeY)-1,len(dist)-1,len(prop)-1
        a=pas*ma.tan(i)
        dist.append(dist[d] + ma.sqrt(a**2 + pas**2))
        prop.append(prop[p]*t)
        ListeX.append(ListeX[x] + a)
        ListeY.append(ListeY[y] + pas)
        k=k+1
        if k<(len(L) - 1) and abs((L[k]/L[k+1])*ma.sin(i))<1:
            t,r=transmission(L[k],L[k+1],i),reflexion(L[k],L[k+1],i)
    if k<(len(L) - 1):
        incidence2=-i
        n2=L[k]
        x,y,d,p=len(ListeX)-1,len(ListeY)-1,len(dist)-1,len(prop)-1
        a=pas*ma.tan(i)
        dist.append(dist[d] + ma.sqrt(a**2 + pas**2))
        if abs((L[k]/L[k+1])*ma.sin(-i))<1:
            r=reflexion(L[k],L[k+1],-i)
        else:
            r=ref_tot(L[k],L[k+1],-i,i)
        prop.append(prop[p]*r)
        ListeX.append(ListeX[x] - a)
        ListeY.append(ListeY[y] - pas)
        for j in range(k,0,-1):
            t=transmission(L[j],L[j-1],i)
            i=ma.asin((n2/L[j-1])*ma.sin(incidence2))
            x,y,d,p=len(ListeX)-1,len(ListeY)-1,len(dist)-1,len(prop)-1
            a=pas*ma.tan(i)
            dist.append(dist[d] + ma.sqrt(a**2 + pas**2))
            prop.append(prop[p]*t)
            ListeX.append(ListeX[x] - a)
            ListeY.append(ListeY[y] - pas)
    return((ListeX,ListeY),(dist,prop),k)
    
    
###2) Étude du rayonnement des antennes


def facteur_reseau_n(angle_début,angle_fin,nb_points,nb_éléments,espacement,l_onde):
# Donne la représentation graphique du facteur de réseau associé à un réseau
# de 'nb_éléments' antennes dipolaires (antenne Yagi) alignées et toutes
# espacées de 'espacement'. Le signal émis par chaque antenne possède la
# longueur d'onde 'l_onde'. Le tracé représente le facteur de réseau dans le
# plan theta=ma.pi/2 (colatitude constante) entre les longitudes 'angle_début'
# et 'angle_fin'. L'intervalle des valeurs de longitude
# [angle_début,angle_fin] est subdivisé en 'nb_points' points.
    ListeA=list(np.linspace(angle_début,angle_fin,nb_points))
    ListeB=[abs(ma.sin(nb_éléments*espacement*ma.pi*ma.sin(a)/l_onde)/ma.sin(espacement*ma.pi*ma.sin(a)/l_onde)) for a in ListeA]
    qte=str(nb_éléments)
    plt.figure()
    plt.plot(ListeA,ListeB)
    plt.title("Diagramme de rayonnement de "+qte+" éléments espacés d'une demie longueur d'onde")
    plt.ylabel("proportion de l'intensité du champ électrique rayonné par une unique antenne")
    plt.xlabel("alpha (radians)")
    plt.show()

    
def F(t,p,nb_éléments):
# Renvoie la puissance rayonnée par une antenne Yagi à 'nb_éléments' éléments
# tous espacés d'une demie longueur d'onde dans la direction repérée par la
# colatitude t et la longitude p en coordonnées sphériques.
    return(ma.sin(t)*(ma.sin((nb_éléments*ma.pi/2)*ma.sin(p))/ma.sin((ma.pi/2)*ma.sin(p))))
    

def diag_ray(fonction,nb_éléments,thetaI,thetaF,phiI,phiF,nP,nT):
# Détermine le diagramme de rayonnement d'une antenne Yagi comportant
# 'nb_éléments' éléments tous espacés d'une demie longueur d'onde. Le
# diagramme de rayonnement est représenté entre les angles 'thetaI' et
# 'thetaF', et 'phiI' et 'phiF' en coordonnées sphériques, chacun de ces
# intervalles étant subdivisés en 'nT' et 'nP' valeurs respectivement.
    T=np.linspace(thetaI,thetaF,nT)
    P=np.linspace(phiI,phiF,nP)
    ListeX,ListeY,ListeZ=[],[],[]
    for t in T:
        for p in P:
            ListeX.append(fonction(t,p,nb_éléments)*ma.sin(t)*ma.cos(p))
            ListeY.append(fonction(t,p,nb_éléments)*ma.sin(t)*ma.sin(p))
            ListeZ.append(fonction(t,p,nb_éléments)*ma.cos(t))
    fig=plt.figure()
    ax=mpl.Axes3D(fig)
    ax.plot(ListeX,ListeY,ListeZ)
    plt.title("Diagramme de rayonnement simulé d'une antenne Yagi à "+str(nb_éléments)+"éléments")
    plt.show()
    
    
# Diagrammes de rayonnement d'antennes Yagi à 20 et 30 éléments espacés d'une
# demie longueur d'onde :
    

diag_ray(F,20,0,ma.pi,-ma.pi/2,ma.pi/2,400,100)
diag_ray(F,30,0,ma.pi,-ma.pi/2,ma.pi/2,400,100)
    

###3) Construction de diagrammes de Voronoï et algorithme de Fortune

## Implémentation d'un tri rapide


def permutation(L,i,j):
    temp=L[j]
    for k in range(j-1,i-1,-1):
        L[k+1]=L[k]
    L[i]=temp
    return L
    
    
def tri_aux_dsc_list_list(L,i,j):
    if i<j:
        pivot=i
        for k in range(i+1,j+1):
            if L[k][1]>L[pivot][1]:
                permutation(L,pivot,k)
                pivot=pivot+1
        tri_aux_dsc_list_list(L,i,pivot-1)
        tri_aux_dsc_list_list(L,pivot+1,j)
        
        
def tri_aux_asc_list(L,i,j):
    if i<j:
        pivot=i
        for k in range(i+1,j+1):
            if L[k]<L[pivot]:
                permutation(L,pivot,k)
                pivot=pivot+1
        tri_aux_asc_list(L,i,pivot-1)
        tri_aux_asc_list(L,pivot+1,j)
        
        
def quick_sort(liste_points):
    n=len(liste_points)
    if n>1:
        tri_aux_asc_list(liste_points,0,n-1)
        
        
## Fonctions auxiliaires constitutives de l'algorithme de Fortune


# On cherche à déterminer le diagramme de Voronoï associé à un ensemble
# d'antennes, supposées ponctuelles, appelées "foyers". Ces foyers sont
# repérés par leurs coordonées dans une liste de listes : 'liste_foyers'. Les
# listes contenues dans 'liste_foyers' sont toutes de dimension 2 de la forme
# [x,y] ou x et y représentent respectivement l'abscisse et l'ordonnée d'un
# foyer. La zone de tracé du diagramme est un plan délimité par les droites
# d'équations :
# x = limx1
# x = limx2
# y = limy1
# y = limy2
# Avec limx1 <= limx2 et limy1 <= limy2.


def distance(x1,y1,x2,y2):
# Détermine la distance entre les points de coordonnées (x1,y1) et (x2,y2).
    return ma.sqrt((x1-x2)**2 + (y1-y2)**2)
    # return np.max(np.abs(x1 - x2), np.abs(y1 - y2))

def copie(liste):
# Renvoie la copie de la liste 'liste'.
    L=[]
    for k in liste:
        L.append(k)
    return L
    
    
def order_y_coord(liste_points):
# 'liste_points' est une liste de listes. Les listes contenues dans
# 'liste_points' sont toutes de dimension 2 de la forme [x,y] ou x et y
# représentent respectivement l'abscisse et l'ordonnée d'un point du plan.
# Renvoie une liste de listes similaire où les points sont classés par
# ordonnée décroissante.
    n=len(liste_points)
    if n>1:
        tri_aux_dsc_list_list(liste_points,0,n-1)
        
    
def parabole(xf,yf,yd,x):
# Pour un foyer de coordonnées (xf,yf) et une droite directrice d'équation
# y=yd, renvoie le réel y tel que le point de coordonnées (x,y) soit
# équidistant du foyer et de la droite directrice (soit du point de
# coordonnées (x,yd)).
    return ((x - xf)**2)/(2*(yf - yd)) + ((yf+yd)/2)
    

def intersec(x1,y1,x2,y2,yd):
# Renvoie l'abscisse du/des point(s) d'intersection des paraboles de même
# directrice y=yd, l'une définie par le foyer 1 de coordonnées (x1,y1) et
# l'aute par le foyer de coordonnées (x2,y2). S'il existe deux points
# d'intersection, renvoie un tuple (a,b) où a<=b.
    if y1==yd:
        return (x1,x1)
    elif y2==yd:
        return (x2,x2)
    A = 1/(2*(y1 - yd)) - 1/(2*(y2 - yd))
    B = x2/(y2 - yd) - x1/(y1 - yd)
    C=(x1**2 + y1**2 - yd**2)/(2*(y1 - yd)) - (x2**2 + y2**2 - yd**2)/(2*(y2 - yd))
    P=np.poly1d([A,B,C])
    rep=np.roots(P)
    if rep[0]>rep[1]:
        return (rep[1],rep[0])
    else:
        return (rep[0],rep[1])
        
        
# On balaie la zone de tracé du diagramme de haut en bas avec la ligne de
# balayage. Cette ligne de balayage d'équation y=yd est également la
# directrice de toutes les paraboles tracées associées aux foyers (les
# antennes) dans la zone de tracé. La courbe formée de la concaténation de
# sections de ces paraboles est appelée 'beachline'.

# Les foyers sont classés par ordonnée décroissante dans la liste globale 'P'.
# On attribue à chaque foyer un numéro correspondant à sa position dans la
# liste P. Ce numéro est représenté par la variable globale 'no_foyer' dans
# les fonctions. Ainsi, le point d'ordonnée la plus élevée se voit attribué le
# numéro 0, puis 1 pour le deuxième point de 'P' (point se trouvant juste en
# dessous du point 0) et ainsi de suite. La ligne de balayage rencontre donc
# les foyers par numéro croissant.

# On caractérise l'état de la beachline pour chaque position yd de la ligne de
# balayage par les deux listes suivantes initialisées comme suit:


etat_arcs=[]
etat_intersec=[]


# 'etat_arcs' caractérise les arcs de paraboles composant la beachline pour
# une position donnée de la ligne de balayage. Ces arcs sont référés par le
# numéro du foyer qui les engendre. Ainsi, si la beachline se compose de
# gauche à droite d'une section de la parabole engendrée par la ligne de
# balayage et le foyer 0, puis d'une section de la parabole engendrée par la
# ligne de balayage et le foyer 1, et enfin d'une section de la parabole
# engendrée par la ligne de balayage et le foyer 0, l'état de la beachline
# sera caractérisé par la liste 'etat_arcs' [0,1,0].

# 'etat_intersec' regroupe les abscisses des points d'intersection des arcs de
# paraboles composant la beachline à une position donnée de la ligne de
# balayage. Ces abscisses sont classées par ordre croissant. Si une valeur
# d'abscisse dans 'etat_intersec' est outre les limites de la zone de tracé du
# diagramme, celle-ci est retirée de 'etat_intersec' (conditions limites, cf
# fonction 'check_lim').

# Mise-à-jour de la beachline lorsque la ligne de balayage rencontre un
# nouveau foyer de coordonnées (x,y)


def maj_nvfoyer(x,y,yd):
# Modifie 'etat_arcs' et 'etat_intersec' lorsque la ligne de balayage y=yd
# rencontre un nouveau foyer de coordonnées (x,y).
    global etat_arcs
    global etat_intersec
    global P
    global no_foyer
    if etat_arcs==[]:
        etat_arcs=[0]
    else:
        k=0
        while len(etat_intersec)>0 and k<len(etat_intersec) and x>etat_intersec[k]:
            k=k+1
        foyer=etat_arcs[k]
        x1,x2=intersec(P[foyer][0],P[foyer][1],x,y,yd)
        if etat_intersec==[]:
            etat_intersec=[x1,x2]
            etat_arcs.append(1)
            etat_arcs.append(0)
        else:
            etat_intersec.insert(k,x2)
            etat_intersec.insert(k,x1)
            etat_arcs.insert(k,no_foyer)
            etat_arcs.insert(k,etat_arcs[k+1])

    
# Les sommets du diagramme de Voronoï calculé sont stockés dans la liste
# 'liste_sommets'. Celle-ci est de la forme suivante :
# [[x,y,L],...] où x et y sont l'abscisse et l'ordonnée respectivement du
# sommet. L est une liste contenant le numéro des foyers dont les cellules
# sont séparées par le sommet. Par exemple, si le sommet est le point de
# concours des arrêtes du diagramme séparant les cellules associées aux foyers
# 1,2, et 3, on aura L = [2,1,3] ou [3,2,1] par exemple (l'ordre n'est pas
# important). Dans le cas où le sommet ne sépare que deux cellules, L est de
# longueur 2 (cf fonction 'check_lim').
    
# Mise-à-jour classique de la beachline dans le cas où la ligne de balayage ne
# rencontre pas de nouveau foyer. La fonction 'maj' redétermine 'etat_arcs' et
# 'etat_intersec' en recalculant l'abscisse des nouveaux points d'intersection
# pour la nouvelle position yd de la ligne de balayage. La fonction détecte
# également les conditions de création de sommets du diagramme de Voronoï. Une
# fois un sommet trouvé, ce dernier est ajouté à 'liste_sommets'.


def maj(yd):
    global etat_intersec
    global etat_arcs
    global P
    global liste_sommets
    k=0
    while k<len(etat_intersec):
        if etat_arcs[k]<etat_arcs[k+1]: # Calcul des nouveaux points
        # d'intersection.
            x1=intersec(P[etat_arcs[k]][0],P[etat_arcs[k]][1],P[etat_arcs[k+1]][0],P[etat_arcs[k+1]][1],yd)[0]
        else:
            x1=intersec(P[etat_arcs[k]][0],P[etat_arcs[k]][1],P[etat_arcs[k+1]][0],P[etat_arcs[k+1]][1],yd)[1]
        if k>0 and etat_intersec[k-1]>=x1: # Condition de création de sommet.
            x=x1
            y=parabole(P[etat_arcs[k]][0],P[etat_arcs[k]][1],yd,x)
            liste_sommets.append([x,y,[etat_arcs[k-1],etat_arcs[k],etat_arcs[k+1]]]) # Inscription du nouveau sommet
            # dans 'liste_sommets'.
            etat_intersec.pop(k-1)
            if etat_arcs[k-1]>etat_arcs[k+1]: # Mise_à-jour de la beachline.
                etat_intersec[k-1]=intersec(P[etat_arcs[k-1]][0],P[etat_arcs[k-1]][1],P[etat_arcs[k+1]][0],P[etat_arcs[k+1]][1],yd)[1]
            else:
                etat_intersec[k-1]=intersec(P[etat_arcs[k-1]][0],P[etat_arcs[k-1]][1],P[etat_arcs[k+1]][0],P[etat_arcs[k+1]][1],yd)[0]
            etat_arcs.pop(k)
        else:
            etat_intersec[k]=x1
        k=k+1
    

# Vérification des conditions limites :


def check_lim(yd,limx1,limx2,limy1,limy2):
# Pour une position 'yd' donnée de la ligne de balayage, et une zone de tracé
# du diagramme délimitée par 'limx1', 'limx2', 'limy1' et 'limy2', vérifie si
# les points d'intersection des arcs de paraboles de la beachline (dont les
# abscisses sont consignées dans 'etat_intersec') sont bien dans la zone de
# tracé. Dans le cas contraire, les points d'intersection outre la zone de
# tracé sont ajoutés comme sommets dans 'liste_sommets'. Ils ne séparent en
# général que deux cellules.
    global etat_intersec
    global etat_arcs
    global P
    global pas
    global liste_sommets
    k=0
    while k<len(etat_intersec):
        L=[etat_arcs[k],etat_arcs[k+1]]
        a=parabole(P[L[0]][0],P[L[0]][1],yd,etat_intersec[k])
        if (a<=limy1 and redondant_lim(L,limy1,limy2,'bas')) or (a>=limy2 and redondant_lim(L,limy1,limy2,'haut')): # Si un point d'intersection de la
        # beachline possède une ordonnée non-comprise entre 'limy1' et
        # 'limy2', on vérifie en plus qu'aucun autre point extérieur à la zone
        # de tracé et séparant les mêmes celleules que le premier point ne
        # soit déjà dans 'liste_sommets' pour éviter d'ajouter des points
        # excédentaires lorsque la beachline est en dehors de la zone de
        # tracé.
            liste_sommets.append([etat_intersec[k],a,L])
        if etat_intersec[k]<=limx1: # Conditions de limites horizontales (en
        # abscisse).
            liste_sommets.append([etat_intersec[k],a,L])
            etat_intersec.pop(0)
            etat_arcs.pop(0)
        elif etat_intersec[k]>=limx2:
            liste_sommets.append([etat_intersec[k],a,L])
            etat_intersec.pop()
            etat_arcs.pop()
        k=k+1
            
    
def a_relier(L1,L2):
# Détermine si deux sommets de 'liste_sommets' sont à relier lors du tracé du
# diagramme (cf fonction 'diagramme_voronoi') ou non en fonction des cellules
# qu'ils séparent. On rapelle la configuration de 'liste_sommets' :
# [[x,y,L],...]. 'L1' et 'L2' correspondent à la liste L dans la configuration
# de 'liste_sommets' et contiennent les numéros des foyers dont les cellules
# associées sont séparées par le premier sommet pour 'L1' et le deuxième pour
# 'L2'.
    k=0
    if len(L1)==2 or len(L2)==2:
        while k<len(L1) and L1[k] in L2:
            k=k+1
        return (k==len(L1) or k==len(L2))
    else:
        i=0
        while k<len(L1):
            if L1[k] in L2:
                i=i+1
            k=k+1
        return(i>=2)
        

def redondant_lim(L,limy1,limy2,position):
# Vérifie si il existe déjà un sommet séparant les cellules dont les numéros
# sont contenus dans L en dehors des limites verticales (ordonnée; 'limy1' et
# 'limy2') de la zone de tracé du diagramme. 'position' est de type str
# ('haut', 'bas') et renseigne la zone de recherche de la fonction soit au
# dessus ou en dessous de la zone de tracé. 'L' est la liste des numéros des
# cellules (ou foyers associés à ces cellules) séparées par le sommet
# considéré.
    global liste_sommets
    A=[k for k in liste_sommets if a_relier(k[2],L)]
    k=0
    if position=='haut':
        while k<len(A) and A[k][1]<limy2:
            k=k+1
        return (k==len(A))
    elif position=='bas':
        while k<len(A) and A[k][1]>limy1:
            k=k+1
        return (k==len(A))
        
        
## Compilation des fonctions auxiliaires
    
    
def sommets(liste_foyers,limx1,limx2,limy1,limy2,nb_positions_ligne_balayage):
# Détermine 'liste_sommets' (les sommets du diagramme de Voronoï) pour
# l'ensemble de foyers 'liste_foyers' dans la zone de tracé délimitée par
# 'limx1', 'limx2', 'limy1', et 'limy2' avec 'nb_positions_ligne_balayage'
# positions successives de la ligne de balayage.
    global pas
    global no_foyer
    global etat_intersec
    global etat_arcs
    global P
    global liste_sommets
    pas=(limy2 - limy1)/(nb_positions_ligne_balayage - 1)
    no_foyer=0
    etat_arcs,etat_intersec=([],[])
    liste_sommets=[]
    order_y_coord(liste_foyers)
    P=copie(liste_foyers)
    for k in range(nb_positions_ligne_balayage + 9000):
        yd=limy2 - k*pas
        maj(yd)
        while no_foyer<len(P) and yd<=P[no_foyer][1]:
            maj_nvfoyer(P[no_foyer][0],P[no_foyer][1],yd)
            no_foyer=no_foyer+1
        check_lim(yd,limx1,limx2,limy1,limy2)
    return liste_sommets
    
            
# Tracé des diagrammes :


def diagramme_voronoi(liste_foyers,limx1,limx2,limy1,limy2,nb_positions_ligne_balayage):
# Exploite les résultats de la fonction 'sommets' et trace le diagramme de
# Voronoï associé à l'ensemble de foyers 'liste_foyers' dans la zone de tracé
# délimitée par 'limx1', 'limx2', 'limy1', et 'limy2' avec
# 'nb_positions_ligne_balayage' positions successives de la ligne de balayage.
    liste_sommets=sommets(liste_foyers,limx1,limx2,limy1,limy2,nb_positions_ligne_balayage)
    plt.figure()
    plt.axis([limx1,limx2,limy1,limy2],'scaled')
    for k in liste_sommets:
        A=[]
        for i in liste_sommets:
            if a_relier(k[2],i[2]): # Appel de la fonction 'a_relier' pour
            # déterminer les sommets à mettre en relation par une arrête du
            # graphe.
                A.append(i)
        for i in A:
            plt.plot([k[0],i[0]],[k[1],i[1]],'b')
    X,Y=([],[])
    for k in liste_foyers:
        X.append(k[0])
        Y.append(k[1])
    plt.title("Diagramme de Voronoï associé à un réseau quelconque d'antennes")
    plt.plot(X,Y,'ro')
    plt.show()


## Diagrammes de Voronoï associés à des configurations aléatoires d'antennes
            
            
# Un exemple pour un réseau quelconque d'antennes :

diagramme_voronoi([[2,5],[3,1],[5,3],[5,6],[6,2],[3,4],[1,1.8],[6,5.5],[2,2.5],[1,6.2],[6.6,6.5],[0.5,2.7],[6.5,0.5],[3.6,0.4],[0.8,0.9],[4.4,3.7],[4,6.4],[1.1,4.5],[6.6,2.8],[0.26,6.7],[2.5,2.2],[6.6,4.66]],0,7,0,7,7000)


# Après l'ajout d'une antenne supplémentaire :

diagramme_voronoi([[2,5],[3,1],[5,3],[5,6],[6,2],[3,4],[1,1.8],[6,5.5],[2,2.5],[1,6.2],[6.6,6.5],[0.5,2.7],[6.5,0.5],[3.6,0.4],[0.8,0.9],[4.4,3.7],[4,6.4],[1.1,4.5],[6.6,2.8],[0.26,6.7],[2.5,2.2],[6.6,4.66],[3.99,4.98]],0,7,0,7,7000)


###4) Cartes cellulaires spécifiques

## Détermination du contour d'une image en noir et blanc


# On cherche à utiliser les diagrammes de rayonnement des antennes dans un
# plan de colatitude constante pour construire les cartes cellulaires adaptées
# aux spécificités des antennes utilisées. Pour cela, l'allure des diagrammes
# de rayonnement est consignée dans un fichier image en noir et blanc. La 
# zone de rayonnement de l'antenne est représentée en blanc tandis que tout le
# reste de l'image est noir. Par détection du contour de la zone blanche
# sur l'image, on peut déterminer et stocker l'allure du diagramme de
# rayonnement de l'antenne ce qui permettra un traitement simple du motif de
# la zone de rayonnement pour la construction des cartes cellulaires.


import os
import cv2 # Module OpenCV

os.chdir('C:\\Users\Gibril\\Documents\\Gib\\TIPE\\Profils antennes\\png')
# Chemin d'accès aux images


im=cv2.imread('antenne_3_lobes(bw).jpg')
im=cv2.imread('antenne_yagi.jpg')
imgray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(imgray,127,255,0)
contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        
cnt1=contours[0]
cnt2=[]
for i in cnt1:
    L=[]
    for j in i:
        L.append(list(j))
    cnt2.append(L)
contour=[]
k=0
while k<len(cnt2):
    contour.append(cnt2[k][0])
    k=k+1
contour=contour+[contour[0]] # 'contour' est une liste de listes de la forme
# [[x,y],...] où x et y représentent l'abscisse et l'ordonnée respectivement
# des points conposant le contour de l'image utilisée.


# Résultat de la détection de contour :

liste1=[k[0] for k in contour]
liste2=[k[1] for k in contour]
plt.figure()
plt.title("Contour identifié pour l'antenne Yagi")
plt.plot(liste1,liste2)
plt.show()


## Transformée de Fourier paramétrique du contour pour obtenir un tracé plus
## précis et continu de celui-ci (diagramme de rayonnement)


# Le contour identifié sous forme de liste de points permet de déterminer les
# courbes paramétriques de l'abscisse et l'ordonnée des points du contour en
# fonction d'un paramètre t compris entre 0 et 2*ma.pi.


h=(2*ma.pi)/(len(contour)-1)
liste_paramx=[[k*h,contour[k][0] - 102] for k in range(len(contour))]
liste_paramy=[[k*h,contour[k][1]] for k in range(len(contour))]
listet=[k*h for k in range(len(contour))]
listex=[k[1] for k in liste_paramx]
listey=[k[1] for k in liste_paramy]
plt.figure()
plt.subplot(2,1,1)
plt.title("Abscisse du contour en fonction du paramètre t")
plt.xlabel("t")
plt.ylabel("abscisse")
plt.plot(listet,listex)
plt.subplot(2,1,2)
plt.title("Ordonnée du contour en fonction du paramètre t")
plt.xlabel("t")
plt.ylabel("ordonnée")
plt.plot(listet,listey)
plt.show()


# Les courbes paramétriques de l'abscisse et l'ordonnée du contour sont
# rendues continues grâce aux fonctions 'abscisse' et 'ordonnee' qui renvoient
# respectivement la valeur de l'abscisse et de l'ordonée du point du contour
# correspondant à la valeur t du paramètre comprise entre 0 et 2*ma.pi. La
# continuité de ces fonctions est nécessaire pour déterminer les coefficients
# de leur décomposition en série de Fourier, calculés par intégration (méthode
# des trapèzes). 


def abscisse(t):
    global liste_paramx
    n=len(liste_paramx)
    if t==0:
        return liste_paramx[0][1]
    else:
        k=0
        while k<n and t>liste_paramx[k][0]:
            k=k+1
        d1=abs(t - liste_paramx[k-1][0])
        d2=abs(t - liste_paramx[k][0])
        D=abs(liste_paramx[k][0] - liste_paramx[k-1][0])
        return (d1*liste_paramx[k][1] + d2*liste_paramx[k-1][1])/D
        

def ordonnee(t):
    global liste_paramy
    n=len(liste_paramy)
    if t==0:
        return liste_paramy[0][1]
    else:
        k=0
        while k<n and t>liste_paramy[k][0]:
            k=k+1
        d1=abs(t - liste_paramy[k-1][0])
        d2=abs(t - liste_paramy[k][0])
        D=abs(liste_paramy[k][0] - liste_paramy[k-1][0])
        return (d1*liste_paramy[k][1] + d2*liste_paramy[k-1][1])/D


# Fonctions pour le calcul des coefficients de la décomposition en série de
# Fourier :


def fcos_n(fonction,n,t): # n est un entier positif
    return fonction(t)*ma.cos(n*t)
    
    
def fsin_n(fonction,n,t): # n est un entier positif
    return fonction(t)*ma.sin(n*t)
    
    
def paramxcos_n(t):
    global k
    return fcos_n(abscisse,k,t)
    
    
def paramxsin_n(t):
    global k
    return fsin_n(abscisse,k,t)
    
    
def paramycos_n(t):
    global k
    return fcos_n(ordonnee,k,t)
    
    
def paramysin_n(t):
    global k
    return fsin_n(ordonnee,k,t)
    

def trapezes(f,a,b,nb_intervalles) :
# Méthode d'intégration des trapèzes de la fonction f sur [a,b] en subdivisant
# [a,b] en 'nb_intervalles' intervalles.
    h=(b-a)/nb_intervalles
    S=0
    for k in range(nb_intervalles):
        S=S+f(a + k*h)+f(a + (k+1)*h)
    return (S*h)/2
    
   
def fourier_transform(contour,nb_termes):
# Pour un contour donné, renvoie (param_x,param_y) = ([x,[a1,b1],[a2,b2],...
# ...,[anb_termes,bnb_termes]],
# [y,[c1,d1],[c2,d2],...,[cnb_termes,dnb_termes]]) tel que le contour soit
# représenté par les équations paramétriques
# x(t) = x + a1*cos(t) + b1*sin(t) + a2*cos(2*t) + b2*sin(2*t) + ... +
# anb_termes*cos(nb_termes*t) + bnb_termes*cos(nb_termes*t) et
# y(t) = y + c1*cos(t) + d1*sin(t) + c2*cos(2*t) + d2*sin(2*t) + ... +
# cnb_termes*cos(nb_termes*t) + dnb_termes*cos(nb_termes*t)
    global liste_paramx
    global liste_paramy
    global k
    h=(2*ma.pi)/(len(contour)-1)
    liste_paramx=[[k*h,contour[k][0]] for k in range(len(contour))]
    liste_paramy=[[k*h,contour[k][1]] for k in range(len(contour))]
    param_x,param_y=[(1/2*ma.pi)*trapezes(abscisse,0,2*ma.pi,1000)],[(1/2*ma.pi)*trapezes(ordonnee,0,2*ma.pi,1000)]
    k=1
    while k<nb_termes + 1:
        param_x.append([(1/ma.pi)*trapezes(paramxcos_n,0,2*ma.pi,1000),(1/ma.pi)*trapezes(paramxsin_n,0,2*ma.pi,1000)])
        param_y.append([(1/ma.pi)*trapezes(paramycos_n,0,2*ma.pi,1000),(1/ma.pi)*trapezes(paramysin_n,0,2*ma.pi,1000)])
        k=k+1
    return (param_x,param_y)
    
    
def fourier_list(param,nb_points):
# 'param' est un tuple du même type que le résultat de 'fourier_transform'.
# Renvoie sous forme d'un tuple de listes le motif dont l'expression
# paramétrique est contenue dans 'param' pour 'nb_points' valeurs du paramètre
# comprises entre 0 et 2*ma.pi.
    param_x,param_y=param
    listet=list(np.linspace(0,2*ma.pi,nb_points))
    listex=[]
    listey=[]
    for t in listet:
        x,y=param_x[0],param_y[0]
        for k in range(1,len(param_x)):
            x=x + param_x[k][0]*ma.cos(k*t) + param_x[k][1]*ma.sin(k*t)
            y=y + param_y[k][0]*ma.cos(k*t) + param_y[k][1]*ma.sin(k*t)
        listex.append(x)
        listey.append(y)
    return (listex,listey)
    
    
def fourier_plot(param,nb_points):
# Même spécification que la fonction précédente. Exploite les résultats de la
# fonction précédente et donne l'allure du motif dont la décomposition
# paramétrique en série de Fourier est contenue dans le tuple 'param'.
    listex,listey=fourier_list(param,nb_points)
    plt.figure()
    plt.plot(listex,listey)
    plt.show()
    
    
def dft_centree_normalisee(contour,nb_termes):
# Pour un contour donné (liste de listes des coordonnées des points du contour
# de l'image), détermine la décomposition en série de Fourier à 'nb_termes'
# termes du contour centré au point de coordonnées (0,0) et contenu dans un
# carré de côté unitaire. Le résultat est du même type de celui de la fonction
# 'fourier_transform'.
    param_x,param_y=fourier_transform(contour,nb_termes)
    param_x[0],param_y[0]=0,0
    x,y=fourier_list((param_x,param_y),250)
    xmax,ymax=x[0],y[0]
    for k in range(1,len(x)):
        if abs(x[k])>xmax:
            xmax=abs(x[k])
        elif abs(y[k])>ymax:
            ymax=abs(y[k])
    for k in range(1,len(param_x)):
        param_x[k]=[(1/xmax)*param_x[k][0],(1/xmax)*param_x[k][1]]
        param_y[k]=[(1/ymax)*param_y[k][0],(1/ymax)*param_y[k][1]]
    return (param_x,param_y)
        

## Compilation des fonctions précédentes


# On compile les fonctions précédentes dans la fonction 'ray_diag' qui renvoie
# directement, pour un fichier image donné, les équations paramétriques du
# diagramme de rayonnement (contour sur l'image) de l'antenne choisie (choix
# de l'image). Ces équations paramétriques sont obtenues par décomposition en
# série de Fourier avec la fonction 'dft_centree_normalisee'.


def ray_diag(image,nb_termes):
# Renvoie un tuple du même type que celui renvoyé par les fonctions
# 'fourier_transform' ou 'dft_centree_normalisee'.
    os.chdir('C:\\Users\Gibril\\Documents\\Gib\\TIPE\\Profils antennes\\png')
    im=cv2.imread(image)
    imgray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh=cv2.threshold(imgray,127,255,0)
    contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt1=contours[0]
    cnt2=[]
    for i in cnt1:
        L=[]
        for j in i:
            L.append(list(j))
        cnt2.append(L)
    contour=[]
    k=0
    while k<len(cnt2):
        contour.append(cnt2[k][0])
        k=k+1
    contour=contour+[contour[0]]
    h=(2*ma.pi)/(len(contour)-1)
    liste_paramx=[[k*h,contour[k][0]] for k in range(len(contour))]
    liste_paramy=[[k*h,contour[k][1]] for k in range(len(contour))]
    return dft_centree_normalisee(contour,nb_termes)
    

# Visualisation des différentes configurations pour deux antennes placées
# différemment. Les diagrammes de rayonnement tracés sont obtenus par
# transformée de Fourier. Les deux antennes utilisées sont une antenne Yagi et
# une antenne à trois lobes principaux d'émission :

antenne1=ray_diag('antenne_yagi.jpg',50)
antenne2=ray_diag('antenne_3_lobes(bw).jpg',40)


# Configuration 1 :

antenne1[0][0],antenne1[1][0]=4,2
antenne2[0][0],antenne2[1][0]=3,4
x_ant1,y_ant1=fourier_list(antenne1,250)
x_ant2,y_ant2=fourier_list(antenne2,250)
plt.figure()
plt.plot(x_ant1,y_ant1)
plt.plot(x_ant2,y_ant2)
plt.plot([4.2,3],[2,3.9],'ro')
plt.title('Configuration 1')
plt.show()

# Configuration 2 :

antenne1[0][0],antenne1[1][0]=6,2
antenne2[0][0],antenne2[1][0]=3,2
x_ant1,y_ant1=fourier_list(antenne1,250)
x_ant2,y_ant2=fourier_list(antenne2,250)
plt.figure()
plt.plot(x_ant1,y_ant1)
plt.plot(x_ant2,y_ant2)
plt.plot([2.98,6.2],[1.93,2],'ro')
plt.title('Configuration 2')
plt.show()

# Configuration 3 :

antenne1[0][0],antenne1[1][0]=2,2
antenne2[0][0],antenne2[1][0]=2,4.1
x_ant1,y_ant1=fourier_list(antenne1,250)
x_ant2,y_ant2=fourier_list(antenne2,250)
plt.figure()
plt.plot(x_ant1,y_ant1)
plt.plot(x_ant2,y_ant2)
plt.plot([1.98,2.2],[4,2],'ro')
plt.title('Configuration 3')
plt.show()


def fourier_exp(param,t):
# 'param' est un tuple de même type que ceux renvoyés par les fonctions
# 'fourier_transform' ou 'dft_centree_normalisee'. 't' est un réel (le
# paramètre compris entre 0 et 2*ma.pi). Retourne les
# coordonnées (x,y) du point du plan associé à la figure décrite par 'param'
# pour la valeur t du paramètre.
    param_x,param_y=param
    x,y=param_x[0],param_y[0]
    for k in range(1,len(param_x)):
        x=x + param_x[k][0]*ma.cos(k*t) + param_x[k][1]*ma.sin(k*t)
        y=y + param_y[k][0]*ma.cos(k*t) + param_y[k][1]*ma.sin(k*t)
    return (x,y)
    

def angle(origine_x,origine_y,xr,yr,xp,yp):
# Détermine le plus petit angle entre le vecteur formé par les points de
# coordonnées (origine_x,origine_y) et (xr,yr) et le vecteur formé par les
# points de coordonnées (origine_x,origine_y) et (xp,yp)
    a=np.array([xr - origine_x,yr - origine_y])
    b=np.array([xp - origine_x,yp - origine_y])
    d1=distance(origine_x,origine_y,xp,yp)
    d2=distance(origine_x,origine_y,xr,yr)
    return np.arccos((np.vdot(a,b))/(d1*d2))
    
 
# Tracé des cartes cellulaires associées aux différentes configuration
# spatiales des deux antennes :


def carte_reseau(position_a1,position_a2,diag_a1,diag_a2,p_init,p_fin,nb_etapes):
# Détermine les frontières entre les cellules définies par chacune des
# antennes sous forme de liste de listes de la forme [[x,y],...] où x et y
# sont respectivement l'abscisse et l'ordonnée des points situés sur la
# frontière entre les deux cellules. La configuration spatiale des deux
# antennes est définie par leurs positions sous forme de tuples de coordonnées
# ('position_a1' et 'position_a2'). Les diagrammes de rayonnement des deux
# antennes (expressions paramétriques et décompositions en série de Fourier)
# sont contenus dans les tuples 'diag_a1' et'diag_a2' au même format que les
# résultats des fonctions 'fourier_transform' ou 'dft_centree_normalisee'. Les
# diagrammes de rayonnement sont étendus de la puissance initiale de
# rayonnement 'p_init' à la puissance maximale de rayonnement 'p_fin' en
# 'nb_etapes' étapes.
    frontiere=[]
    xr,yr=fourier_exp(diag_a2,0)
    a1_x,a1_y=copie(diag_a1[0]),copie(diag_a1[1])
    a2_x,a2_y=copie(diag_a2[0]),copie(diag_a2[1])
    a1_x.pop(0)
    a2_x.pop(0)
    a1_y.pop(0)
    a2_y.pop(0)
    a1_x,a1_y=np.array(a1_x),np.array(a1_y)
    a2_x,a2_y=np.array(a2_x),np.array(a2_y)
    x1,y1=position_a1
    x2,y2=position_a2
    pas=abs(p_fin - p_init)/(nb_etapes - 1)
    listet=np.linspace(0,2*ma.pi,700)
    interieur=False
    for k in range(nb_etapes):
        a1_x,a1_y=(p_init + k*pas)*a1_x,(p_init + k*pas)*a1_y
        a2_x,a2_y=(p_init + k*pas)*a2_x,(p_init + k*pas)*a2_y
        for t in listet:
            if t==0:
                xr,yr=fourier_exp(([x2]+list(a2_x),[y2]+list(a2_y)),0)
                x,y=fourier_exp(([x1]+list(a1_x),[y1]+list(a1_y)),0)
                th=angle(x2,y2,xr,yr,x,y)
                th_c=2*ma.pi - th
                xi,yi=fourier_exp(([x2]+list(a2_x),[y2]+list(a2_y)),th)
                xi_c,yi_c=fourier_exp(([x2]+list(a2_x),[y2]+list(a2_y)),th_c)
                if distance(x,y,xi_c,yi_c)<distance(x,y,xi,yi):
                    xi,yi=xi_c,yi_c
                if distance(x,y,x2,y2)<=distance(x2,y2,xi,yi):
                    interieur=True
                elif distance(x,y,x2,y2)>=distance(x2,y2,xi,yi):
                    interieur=False
            x,y=fourier_exp(([x1]+list(a1_x),[y1]+list(a1_y)),t)
            th=angle(x2,y2,xr,yr,x,y)
            th_c=2*ma.pi - th
            xi,yi=fourier_exp(([x2]+list(a2_x),[y2]+list(a2_y)),th)
            xi_c,yi_c=fourier_exp(([x2]+list(a2_x),[y2]+list(a2_y)),th_c)
            if distance(x,y,xi_c,yi_c)<distance(x,y,xi,yi):
                xi,yi=xi_c,yi_c
            if distance(x,y,x2,y2)<=distance(x2,y2,xi,yi) and interieur==False:
                frontiere.append([x,y])
                interieur=True
            elif distance(x,y,x2,y2)>=distance(x2,y2,xi,yi) and interieur==True:
                frontiere.append([x,y])
                interieur=False
    return frontiere
    
    
## Cartes cellulaires associées aux trois configurations :


# Configuration 1 :

frontiere_a=carte_reseau((4,2),(3,4),antenne1,antenne2,1,1.09,50)

front_x=[k[0] for k in frontiere_a]
front_y=[k[1] for k in frontiere_a]
plt.figure()
plt.plot(front_x,front_y,'.')
plt.plot([4.2,3],[2,3.9],'ro')
plt.title('Carte cellulaire pour la configuration 1')
plt.show()

# Configuration 2 :

frontiere2=carte_reseau((6,2),(3,2),antenne1,antenne2,1,1.07,100)

front_x=[k[0] for k in frontiere2]
front_y=[k[1] for k in frontiere2]
plt.figure()
plt.plot(front_x,front_y,'.')
plt.plot([2.98,6.2],[1.93,2],'ro')
plt.title('Carte cellulaire pour la configuration 2')
plt.show()

# Configuration 3 :

frontiere3=carte_reseau((2,2),(2,4.1),antenne1,antenne2,1,1.02,200)

front_x=[k[0] for k in frontiere3]
front_y=[k[1] for k in frontiere3]
plt.figure()
plt.plot(front_x,front_y,'.')
plt.plot([1.98,2.2],[4,2],'ro')
plt.title('Carte cellulaire pour la configuration 3')
plt.show()

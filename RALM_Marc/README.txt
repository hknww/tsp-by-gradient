Explications des différents éléments :

 - data (dossier) :
     - contient des instances de différentes tailles, déjà résolues avec GUROBI
     - les instances sont triées par tailles dans différents dossiers
     - les plus petites tailles disposes de 1000 instances chacunes, les tailles suivantes ne disposent que de 50 instances

 - instance_generator (py ou ipynb):
     - permet de générer une grandes quantitées d'instances et de les résoudres
     - il est présent dans ce rendu afin d'être conservé, et de générer d'autres instances si besoin

 - test_ralm_poc (py ou ipynb) :
     - Preuve de concept de la résolution RALM sur le problème suivant : min x_1 + x_2 s.t. x_1² + x_2² = 32

 - test_ralm_v4 (py ou ipynb) :
     - Fichier principale de résolution par RALM
     - Contient la fonction de résolution
     - Contient la fonction d'analyse de résultat sur un ensemble d'instances

 - tsp_gradient_outils (py) :
     - fonction outils utiles lors de nos travaux
     - Contient la classe associé à un problème de TSP
     - Contient les restes d'une classe de résolution par descente de tsp_gradient_outils

 - test.csv et test_size_8.csv :
     - deux instances utilisées dans différents tests
     - test.csv est de taille 6
     - test_size_8.csv est de taille 8
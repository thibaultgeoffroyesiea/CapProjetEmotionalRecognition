# ML_model_implementation

## Pour le dataset avec les coordonnées:
 loader.read_dataset('Database.csv',class_path='labelsDefault.txt',separator=',')
 loader.X=loader.normalize([299,435])

## Pour le cas où la séparation des classes est différentes:
* Pour faire la séparation, groupe émotion primaire+goupes d'émotion complexe individuel:
  *   class_to_group = {1:[1,2,3,4,5,6,7],2:[8],3:[9],4:[10],5:[11],6:[12],7:[13],8:[14],9:[15],10:[16],11:[17],12:[18],13:[19],14:[20],15:[21],16:[22]}
* Pour faire la séparation, groupe émotion primaire+groupes d'émotion complexe similaire:
  *   class_to_group = {1:[1,2,3,4,5,6,7],2:[8,9],3:[10,11,12,13],4:[14,15,16],5:[17,18],6:[19],7:[20,21,22]}

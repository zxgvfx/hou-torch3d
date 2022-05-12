def getGeomtryByKind(stage,
                    listPath,
                    type = 'component'):
    if stage.GetMetadata('kind') != type and len(stage.GetAllChildren()) != 0:
        for child in stage.GetAllChildren():
            getGeomtryByKind(child,listPath, type)
    elif stage.GetMetadata('kind') == type:
        listPath.append(stage)

def getLightByKind(stage,
                    listPath,
                    type = 'Light'):
    if type not in stage.GetTypeName() and len(stage.GetAllChildren()) != 0:
        for child in stage.GetAllChildren():
            getLightByKind(child,listPath,type)
    elif type in stage.GetTypeName():
        listPath.append(stage)    

def getLPE(listPath,type):   
    if type == 'geo':
        attName = 'primvars:karma:object:lpetag'
    if type == 'light':
        attName = 'karma:light:lpetag'
    listLPE = [path.GetAttribute(attName).Get() for path in listPath]
    return listLPE

stage = hou.pwd().inputs()[0].stage().GetPseudoRoot()
geoListPath = []
lightListPath = []
getGeomtryByKind(stage,geoListPath)
getLightByKind(stage,lightListPath)

iteration_geo = int(hou.contextOption("ITERATIONGEO"))
iteration_light = int(hou.contextOption("ITERATIONLIGHT"))
geoLPE = getLPE(geoListPath,'geo')[iteration_geo]
lightLPE = getLPE(lightListPath,'light')[iteration_light]
#<L.'light3'>
return "C<..'{}'.>.*<L.'{}'>".format(geoLPE,lightLPE)

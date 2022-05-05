node = hou.pwd()
def getNetworkNodeName(node):
    parent = node.parent()
    if parent.isNetwork():
        return parent
    else:
        getNetworkNodeName(parent)
def getRedNode(node):
    children =[i for i in node.children() if i.color() == hou.Color((1,0,0)) and i.type().name() == 'null']
    return children
def createSopNode(nodeList):
    root = hou.node('/obj')
    for i in nodeList:
        sop = root.createNode('geo',i.name())
        objm = sop.createNode('object_merge',i.name())
        objm.parm('objpath1').set(i.path())
        objm.parm('xformtype').set('local')
    
    
networkNode = getNetworkNodeName(node)
nullNode = getRedNode(networkNode)
createSopNode(nullNode)

import hou
import re
keyDict = {'filecache':['file'],
                'ifd':['vm_dcmfilename','vm_picture'],
                'alembic':['fileName'],
                'alembicarchive':['fileName'],
                'matnet':['texture'],
                'mat':['texture']}
objTypeList = ['filecache','ifd','alembic','alembicarchive']
matTypeList = ['matnet','mat']

def changePath(parm,newF = 'O'):
        parmEval = parm.unexpandedString()
        search = re.compile(r'\`(.+?)\`').search(parmEval)
        if parmEval != '':
            if '$HIP' in parmEval or parm.isAtDefault() or '$HFS' in parmEval:
                pass
            elif search :
                search1 = re.compile(r'\"(.+?)\"').search(search.group())
                parmpath = '{}/{}'.format(parm.node().path(),search1.group()[1:-1])
                parm = hou.parm(parmpath)
                if parm:
                    if '$HIP' in parm.unexpandedString():
                        pass
                    else:
                        print('节点{}:\n参数面板{}\n路径{}\n更改O盘成功'.format(parm.node().path(),parm.name(),parm.unexpandedString()))
                        newStr = 'O' + parm.unexpandedString()[1:]
                        parm.set(newStr)
            elif parm.unexpandedString()[0].upper()!='O':
                print('节点{}:\n参数面板{}\n路径{}\n更改O盘成功'.format(parm.node().path(),parm.name(),parm.unexpandedString()))
                newStr = 'O' + parm.unexpandedString()[1:]
                parm.set(newStr)                
                
def checkNodePath(node,keyDict,type):
    if type == 'matnet' or type == 'mat':
        for k in keyDict[type]:
            parms = node.parms()
            for parm in parms:
                if k in parm.name() and parm.parmTemplate().dataType().name() == 'String':
                    changePath(parm)
    else:
        for k in keyDict[type]:
            parm = node.parm(k)
            changePath(parm)
    
def getTypeNode(node,t,keyDict):
    if node.type().name()==t:
        checkNodePath(node,keyDict,t)
    elif len(node.children()) > 0 and node.isLockedHDA()==False:
            for child in node.children():
                getTypeNode(child,t,keyDict)

node = hou.node('/')
for matType in matTypeList:
    getTypeNode(node,matType,keyDict)
for objType in objTypeList:
    getTypeNode(node,objType,keyDict)
print('脚本执行完毕！')

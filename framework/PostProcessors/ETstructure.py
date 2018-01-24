from __future__ import division, print_function , unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)

#Internal Modules---------------------------------------------------------------
import MessageHandler
from utils import utils
from utils import xmlUtils as xmlU
#Internal Modules End-----------------------------------------------------------

#External Modules---------------------------------------------------------------
import numpy as np
import xml.etree.ElementTree as ET
import copy
import itertools
from collections import OrderedDict
#External Modules End-----------------------------------------------------------

#class ETstructure(MessageHandler.MessageUser):
class ETstructure():
    
    #def __init__(self, expand, inputs, messageHandler):
    def __init__(self, expand, inputs):
        """
          This method executes the postprocessor action.
          @ In,  inputs, list, list of file objects
          @ Out, None
        """
        self.expand = expand
        #self.messageHandler = messageHandler
        ### Check for link to other ET
        links = []
        sizes=(len(inputs),len(inputs))
        connectivityMatrix = np.zeros(sizes)
        listETs=[]
        listRoots=[]
    
        for file in inputs:
            eventTree = ET.parse(file.getPath() + file.getFilename())
            listETs.append(eventTree.getroot().get('name'))
            listRoots.append(eventTree.getroot())
        links = self.createLinkList(listRoots)
    
        if len(inputs)>0:
            rootETID = self.checkETstructure(links,listETs,connectivityMatrix)
    
        if len(links)>=1 and len(inputs)>1:
            finalAssembledTree = self.analyzeMultipleET(inputs,links,listRoots,listETs,rootETID)
            self.pointSet = self.analyzeSingleET(finalAssembledTree)
    
        if len(links)==0 and len(inputs)>1:
            raise IOError('Multiple ET files have provided but they are not linked')
    
        if len(links)>1 and len(inputs)==1:
            raise IOError('A single ET files has provided but it contains a link to an additional ET')
    
        if len(links)==0 and len(inputs)==1:
            eventTree = ET.parse(inputs[0].getPath() + inputs[0].getFilename())
            self.pointSet = self.analyzeSingleET(eventTree.getroot())    
        
    def solve(self,combination):
        combinationArray=np.zeros(len(self.variables))
        outcome = -1
        for index, var in enumerate(self.variables):
            combinationArray[index] = combination[var]
        for row in self.pointSet:
            if self.pointSet[:len(self.variables)] == combinationArray:
                outcome = self.pointSet[:, -1]
        return outcome
    
    def returnDict(self):
        outputDict = {}
        outputDict['inputs'] = {}
        outputDict['outputs'] = {}
        for index, var in enumerate(self.variables):
            outputDict['inputs'][var] = self.pointSet[:, index]
        outputDict['outputs']['sequence'] = self.pointSet[:, -1]
    
        return outputDict, self.variables
        
    def createLinkList(self,listRoots):
        """
          This method identifies the links among ETs. It saves such links in the variable links.
          Note that this method overwrites such variable when it is called. This is because the identification
          of the links needs to be computed from scratch after every merging step since the structure of ETs has changed.
          The variable links=[dep1,...,depN] is a list of connections where each connection dep is a
          dictionary as follows:

              dep.keys =
                         * link_seqID  : ID of the link in the master ET
                         * ET_slave_ID : ID of the slave ET that needs to be copied into the master ET
                         * ET_master_ID: ID of the master ET;

          The slave ET is merged into the master ET; note the master ET contains the link in at least one
          <define-sequence> node:

             <define-sequence name="Link-to-LP">
               <event-tree name="Link-to-LP-Event-Tree"/>
             </define-sequence>

          @ In,  listRoots, list, list containing the root of all ETs
          @ Out, linkList, list, list containing the links among the ETs
        """
        linkList = []
        for root in listRoots:
            links, seqID = self.checkLinkedTree(root)
            if len(links) > 0:
                for idx, val in enumerate(links):
                    dep = {}
                    dep['link_seqID'] = copy.deepcopy(seqID[idx])
                    dep['ET_slave_ID'] = copy.deepcopy(val)
                    dep['ET_master_ID'] = copy.deepcopy(root.get('name'))
                    linkList.append(dep)
        return linkList

    def checkETstructure(self,links,listETs,connectivityMatrix):
        """
          This method checks that the structure of the ET is consistent. In particular, it checks that only one root ET
          and at least one leaf ET is provided. As an example consider the following ET structure:
             ET1 ----> ET2 ----> ET3
              |------> ET4 ----> ET5
          Five ETs have been provided, ET1 is the only root ET while ET3 and ET5 are leaf ET.
          @ In, listETs, list, list containing the ID of the ETs
          @ In, connectivityMatrix, np.array, matrix containing connectivity mapping
          @ Out, rootETID, xml.etree.Element, root of the main ET
        """

        # each element (i,j) of the matrix connectivityMatrix shows if there is a connection from ET_i to ET_j:
        #   * 0: no connection from i to j
        #   * 1: a connection exists from i to j
        for link in links:
            row = listETs.index(link['ET_master_ID'])
            col = listETs.index(link['ET_slave_ID'])
            connectivityMatrix[row,col]=1.0

        # the root ETs are charaterized by a column full of zeros
        # the leaf ETs are charaterized by a row full of zeros
        zeroRows    = np.where(~connectivityMatrix.any(axis=1))[0]
        zeroColumns = np.where(~connectivityMatrix.any(axis=0))[0]

        if len(zeroColumns)>1:
            raise IOError('Multiple root ET')
        if len(zeroColumns)==0:
            raise IOError('No root ET')
        if len(zeroColumns)==1:
            rootETID = listETs[zeroColumns]
            print("ETImporter Root ET: " + str(rootETID))

        leafs = []
        for index in np.nditer(zeroRows):
            leafs.append(listETs[index])
        print("ETImporter leaf ETs: " + str(leafs))

        return rootETID

    def analyzeMultipleET(self,inputs,links,listRoots,listETs,rootETID):
        """
          This method executes the analysis of the ET if multiple ETs are provided. It merge all ETs onto the root ET
          @ In, input, list, list of file objects
          @ In, links, list, list containing the links among the ETs
          @ In, listRoots, list containing the root of all ETs
          @ In, listETs, list, list containing the ID of the ETs
          @ In, rootETID, xml.etree.Element, root of the main ET
          @ Out, xmlNode, xml.etree.Element, root of the assembled root ET
        """
        # 1. for all ET check if it contains SubBranches
        ETset = []
        for fileID in inputs:
            eventTree = ET.parse(fileID.getPath() + fileID.getFilename())
            root = self.checkSubBranches(eventTree.getroot())
            ETset.append(root)

        # 2. loop on the dependencies until it is empty
        while len(links)>0:
            for link in links:
                indexMaster = listETs.index(link['ET_master_ID'])
                indexSlave  = listETs.index(link['ET_slave_ID'])
                mergedTree  = self.mergeLinkedTrees(listRoots[indexMaster],listRoots[indexSlave],link['link_seqID'])

                listETs.pop(indexMaster)
                listRoots.pop(indexMaster)

                listETs.append(link['ET_master_ID'])
                listRoots.append(mergedTree)

                links = self.createLinkList(listRoots)

        indexRootET = listETs.index(rootETID)
        return listRoots[indexRootET]

    def analyzeSingleET(self,masterRoot):
        """
          This method executes the analysis of the ET if a single ET is provided.
          @ In,  masterRoot, xml.etree.Element, root of the ET
          @ Out, outputDict, dict, dictionary containing the pointSet data
        """
        root = self.checkSubBranches(masterRoot)

        ## These outcomes will be encoded as integers starting at 0
        outcomes = []
        ## These variables will be mapped into an array where there index
        self.variables = []
        values = {}
        for node in root.findall('define-functional-event'):
            event = node.get('name')
            ## First, map the variable to an index by placing it in a list
            self.variables.append(event)
            ## Also, initialize the dictionary of values for this variable so we can
            ## encode them as integers as well
            values[event] = []
            ## Iterate through the forks that use this event and gather all of the
            ## possible states
            for fork in self.findAllRecursive(root.find('initial-state'), 'fork'):
                if fork.get('functional-event') == event:
                    for path in fork.findall('path'):
                        state = path.get('state')
                        if state not in values[event]:
                            values[event].append(state)

        ## Iterate through the sequences and gather all of the possible outcomes
        ## so we can numerically encode them latter
        for node in root.findall('define-sequence'):
            outcome = node.get('name')
            if outcome not in outcomes:
                outcomes.append(outcome)
        etMap = self.returnMap(outcomes, root.get('name'))
        print("ETImporter variables identified: " + str(format(self.variables)))

        d = len(self.variables)
        n = len(self.findAllRecursive(root.find('initial-state'), 'sequence'))
        pointSet = -1 * np.ones((n, d + 1))
        rowCounter = 0
        for node in root.find('initial-state'):
            newRows = self.constructPointDFS(node, self.variables, values, etMap, pointSet, rowCounter)
            rowCounter += newRows

        if self.expand:
            pointSet = self.expandPointSet(pointSet)
            
        return pointSet

    def expandPointSet(self,pointSet):
        """
          This method performs a full-factorial expansion of the ET: if a branch contains a -1 element this method
          duplicate the branch; each duplicated branch contains element values equal to +1 and 0.
          @ In,  pointSet, np.array, original point set
          @ Out, pointSet, np.array, expanded point set
        """
        for col in range(pointSet.shape[1]):
            indexes = np.where(pointSet[:,col] == -1)[0]
            if indexes.size>0:
                for idx in indexes:
                    rowToBeAdded = copy.deepcopy(pointSet[idx,:])
                    rowToBeAdded[col] = +1
                    pointSet = np.vstack([pointSet,rowToBeAdded])
                    pointSet[idx,col] = 0          
                    #self.expandRow(pointSet,idx,col)
        return pointSet

    def checkLinkedTree(self, root):
        """
          This method checks if the provided root of the ET contains links to other ETs.
          This occurs if a <define-sequence> node contains a <event-tree> sub-node:
              <define-sequence name="Link-to-LP">
                <event-tree name="Link-to-LP-Event-Tree"/>
              </define-sequence>
          The name of the <event-tree> is the link to the ET defined as follows:
              <define-event-tree name="Link-to-LP-Event-Tree">

          @ In,  root, xml.etree.Element, root of the root ET
          @ Out, dependencies, list, ID of the linked ET (e.g., Link-to-LP-Event-Tree)
          @ Out, seqID, list, ID of the link in the root ET (e.g., Link-to-LP)
        """
        dependencies = []
        seqID        = []

        for node in root.findall('define-sequence'):
            for child in node.getiterator():
                if 'event-tree' in child.tag:
                    dependencies.append(child.get('name'))
                    seqID.append(node.get('name'))
        return dependencies, seqID

    def mergeLinkedTrees(self,rootMaster,rootSlave,location):
        """
          This method merged two ET; it merges a slave ET onto the master ET. Note that slave ET can be copied
          in multiple branches of the master ET.
          @ In,  rootMaster, xml.etree.Element, root of the master ET
          @ In,  rootSlave, xml.etree.Element,  root of the slave ET
          @ In,  location, string, ID of the link that identifies the branches of the master ET that are linked to the slave ET
          @ Out, rootMaster, xml.etree.Element, root of the master ET after the merging process has completed
        """
        # 1. copy define-functional-event block
        for node in rootSlave.findall('define-functional-event'):
            rootMaster.append(node)
        # 2. copy define-sequence block
        for node in rootSlave.findall('define-sequence'):
            rootMaster.append(node)
        # 3. remove the <define-sequence> that points at the "location"
        for node in rootMaster.findall('define-sequence'):
            if node.get('name') == location:
                rootMaster.remove(node)
        # 4. copy slave ET into master ET
        for node in rootMaster.findall('.//'):
            if node.tag == 'path':
                for subNode in node.findall('sequence'):
                    linkName = subNode.get('name')
                    if linkName == location:
                        node.append(rootSlave.find('initial-state').find('fork'))
                        node.remove(subNode)
        return copy.deepcopy(rootMaster)

    def checkSubBranches(self,root):
        """
          This method checks if the provided ET contains sub-branches (i.e., by-pass).
          This occurs if the node <define-branch> is provided.
          As an example:
              <define-branch name="sub-tree1">
          defines a branch that is linked to multiple ET sequences:
              <path state="0">
                  <branch name="sub-tree1"/>
              </path>
          The scope of this method is to copy the <define-branch> into the sequences of the
          ET that are linked to <define-branch>
          @ In,  root, xml.etree.Element, root of the ET
          @ Out, root, xml.etree.Element, root of the processed ET
        """
        eventTree = root.findall('initial-state')

        if len(eventTree) > 1:
            print('ETImporter: more than one initial-state identified')
        ### Check for sub-branches
        subBranches = {}
        for node in root.findall('define-branch'):
            subBranches[node.get('name')] = node.find('fork')
            print("ETImporter branch identified: " + str(node.get('name')))
        if len(subBranches) > 0:
            for node in root.findall('.//'):
                if node.tag == 'path':
                    for subNode in node.findall('branch'):
                        linkName = subNode.get('name')
                        if linkName in subBranches.keys():
                            node.append(subBranches[linkName])
                        else:
                            raise IOError(' ETImporter: branch ' + str(
                  linkName) + ' linked in the ET is not defined; available branches are: ' + str(
                      subBranches.keys()))

        for child in root:
            if child.tag == 'branch':
                root.remove(child)

        return root

    def returnMap(self,outcomes,name):
        """
          This method returns a map if the ET contains symbolic sequences.
          This is needed since since RAVEN requires numeric values for sequences.
          @ In,  outcomes, list, list that contains all the sequences IDs provided in the ET
          @ In,  name, string, name of the ET
          @ Out, etMap, dict, dictionary containing the map
        """
        # check if outputMap contains string ID for  at least one sequence
        # if outputMap contains all numbers then keep the number ID
        allFloat = True
        for seq in outcomes:
            try:
                float(seq)
            except ValueError:
                allFloat = False
                break
        etMap = {}
        if allFloat == False:
            # create an integer map, and create an integer map file
            root = ET.Element('map')
            root.set('Tree', name)
            for seq in outcomes:
                etMap[seq] = outcomes.index(seq)
                ET.SubElement(root, "sequence", ID=str(outcomes.index(seq))).text = str(seq)
            fileID = name + '_mapping.xml'
            updatedTreeMap = ET.ElementTree(root)
            xmlU.prettify(updatedTreeMap)
            updatedTreeMap.write(fileID)
        else:
            for seq in outcomes:
                etMap[seq] = utils.floatConversion(seq)
        return etMap

    def findAllRecursive(self, node, element):
        """
          A function for recursively traversing a node in an elementTree to find
          all instances of a tag.
          Note that this method differs from findall() since it goes for all nodes,
          subnodes, subsubnodes etc. recursively
          @ In, node, ET.Element, the current node to search under
          @ In, element, str, the string name of the tags to locate
          @ InOut, result, list, a list of the currently recovered results
        """
        result=[]
        for elem in node.iter(tag=element):
            result.append(elem)
        return result

    def constructPointDFS(self, node, inputMap, stateMap, outputMap, X, rowCounter):
        """
          Construct a "sequence" using a depth-first search on a node, each call
          will be on a fork except in the base case which will be called on a
          sequence node. The recursive case will traverse into a path node, thus
          path nodes will be "skipped" in the call stack as one level of paths
          will be processed per recursive call in order to set one of the columns
          of X for the row identified by rowCounter.
          @ In, node, ET.Element, the current node to process
          @ In, inputMap, list, a map for converting string variable names into
                sequential non-negative integers that can be used to index X
          @ In, stateMap, dict, a map similar to above, but instead converts the
                possible states for each event (variable) into non-negative
                integers
          @ In, outputMap, list, a map for converting string outcome values into
                non-negative integers
          @ In, X, np.array, data object to populate with values
          @ In, rowCounter, int, the row we are currently editing in X
          @ Out, offset, int, the number of rows of X this call has populated
        """

        # Construct point
        if node.tag == 'sequence':
            col = X.shape[1]-1
            outcome = node.get('name')
            val = outputMap[outcome]
            X[rowCounter, col] = val
            rowCounter += 1
        elif node.tag == 'fork':
            event = node.get('functional-event')
            col = inputMap.index(event)

            for path in node.findall('path'):
                state = path.get('state')
                if   state == 'failure':
                    val = '+1'
                elif state =='success':
                    val = '0'
                else:
                    val = stateMap[event].index(state)
                ## Fill in the rest of the data as the recursive nature will only
                ## fill in the details under this branch, later iterations will
                ## correct lower rows if a path does change
                X[rowCounter, col] = val
                for fork in path.getchildren():
                    newCounter = self.constructPointDFS(fork, inputMap, stateMap, outputMap, X, rowCounter)
                    for i in range(newCounter-rowCounter):
                        X[rowCounter+i, col] = val
                    rowCounter = newCounter

        return rowCounter
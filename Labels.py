# -*- coding: utf-8 -*-
"""

@author: ramon, bojana
"""
import re
import numpy as np
import ColorNaming as cn
from skimage import color
import KMeans as km

def NIUs():
    return 1458383, 1458626, 3333333

def loadGT(fileName):
    """@brief   Loads the file with groundtruth content

    @param  fileName  STRING    name of the file with groundtruth

    @return groundTruth LIST    list of tuples of ground truth data
                                (Name, [list-of-labels])
    """

    groundTruth = []
    fd = open(fileName, 'r')
    for line in fd:
        splitLine = line.split(' ')[:-1]
        labels = [''.join(sorted(filter(None,re.split('([A-Z][^A-Z]*)',l)))) for l in splitLine[1:]]
        groundTruth.append( (splitLine[0], labels) )

    return groundTruth


def evaluate(description, GT, options):
    """@brief   EVALUATION FUNCTION
    @param description LIST of color name lists: contain one lsit of color labels for every images tested
    @param GT LIST images to test and the real color names (see  loadGT)
    @options DICT  contains options to control metric, ...
    @return mean_score,scores mean_score FLOAT is the mean of the scores of each image
                              scores     LIST contain the similiraty between the ground truth list of color names and the obtained
    """
    scores = [0] * len(description)
    for i in range (0, len(description)):
        scores[i] = similarityMetric(description[i], GT[i], options)
    return sum(scores)/len(description), scores



def similarityMetric(Est, GT, options):
    """@brief   SIMILARITY METRIC
    @param Est LIST  list of color names estimated from the image ['red','green',..]
    @param GT LIST list of color names from the ground truth
    @param options DICT  contains options to control metric, ...
    @return S float similarity between label LISTs
    """

    if options == None:
        options = {}
    if not 'metric' in options:
        options['metric'] = 'basic'

#########################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#########################################################
    comptador = 0
    if options['metric'].lower() == 'basic':
        for i in Est:
            if i in GT[1]:
                comptador = comptador + 1
        return comptador / len(Est)

    else:
        return 0

def getLabels(kmeans, options):
    """@brief   Labels all centroids of kmeans object to their color names

    @param  kmeans  KMeans      object of the class KMeans
    @param  options DICTIONARY  options necessary for labeling

    @return colors  LIST    colors labels of centroids of kmeans object
    @return ind     LIST    indexes of centroids with the same color label
    """
    colors = []
    unics=[]
    for i, c in enumerate(kmeans.centroids):
        sc=np.flip(np.argsort(c))
        color=""
        if c[sc[0]]>options['single_thr']:
           color=cn.colors[sc[0]]
        else:
           colorSort=sorted([cn.colors[sc[0]], cn.colors[sc[1]]])
           color=colorSort[0]+colorSort[1]
        if color in colors:
            unics[colors.index(color)].append(i)
        else:
            colors.append(color)
            unics.append([i])

    return colors, unics


def processImage(im, options):
    """@brief   Finds the colors present on the input image

    @param  im      LIST    input image
    @param  options DICTIONARY  dictionary with options

    @return colors  LIST    colors of centroids of kmeans object
    @return indexes LIST    indexes of centroids with the same label
    @return kmeans  KMeans  object of the class KMeans
    """

#########################################################
##  YOU MUST ADAPT THE CODE IN THIS FUNCTIONS TO:
##  1- CHANGE THE IMAGE TO THE CORRESPONDING COLOR SPACE FOR KMEANS
##  2- APPLY KMEANS ACCORDING TO 'OPTIONS' PARAMETER
##  3- GET THE NAME LABELS DETECTED ON THE 11 DIMENSIONAL SPACE
#########################################################

##  1- CHANGE THE IMAGE TO THE CORRESPONDING COLOR SPACE FOR KMEANS
    if options['colorspace'].lower() == 'ColorNaming'.lower():
        im = cn.ImColorNamingTSELabDescriptor(im)
    elif options['colorspace'].lower() == 'RGB'.lower():
        pass
    elif options['colorspace'].lower() == 'Lab'.lower():
        im = color.rgb2lab(im)
    elif options['colorspace'].lower() == 'HED'.lower():
        im = color.rgb2hed(im)
    elif options['colorspace'].lower() == 'HSV'.lower():
        im = color.rgb2hsv(im)
    '''
    elif options['colorspace'].lower() == 'opponent'.lower():
        im = color.rgb2lab(im)
    elif options['colorspace'].lower() == 'HSL'.lower():
        im = color.rgb2(im)
    elif options['colorspace'].lower() == 'Lab'.lower():
        im = color.rgb2lab(im)
    '''


##  2- APPLY KMEANS ACCORDING TO 'OPTIONS' PARAMETER
    if options['K']<2: # find the bes K
        kmeans = km.KMeans(im, 0, options)
        kmeans.bestK()
    else:
        kmeans = km.KMeans(im, options['K'], options)
        kmeans.run()

##  3- GET THE NAME LABELS DETECTED ON THE 11 DIMENSIONAL SPACE
    if options['colorspace'].lower() == 'Lab'.lower():
        kmeans.centroids = cn.ImColorNamingTSELabDescriptor(color.lab2rgb(kmeans.centroids.reshape(1,len(kmeans.centroids),3)).reshape(len(kmeans.centroids),3))
    elif options['colorspace'].lower() == 'HED'.lower():
        kmeans.centroids = cn.ImColorNamingTSELabDescriptor(color.hed2rgb(kmeans.centroids.reshape(1,len(kmeans.centroids),3)).reshape(len(kmeans.centroids),3))
    elif options['colorspace'].lower() == 'HSV'.lower():
        kmeans.centroids = cn.ImColorNamingTSELabDescriptor(color.hsv2rgb(kmeans.centroids.reshape(1,len(kmeans.centroids),3)).reshape(len(kmeans.centroids),3))
    elif options['colorspace'].lower() == 'RGB'.lower():
        kmeans.centroids = cn.ImColorNamingTSELabDescriptor(kmeans.centroids)

#########################################################
##  THE FOLLOWING 2 END LINES SHOULD BE KEPT UNMODIFIED
#########################################################
    colors, which = getLabels(kmeans, options)
    return colors, which, kmeans

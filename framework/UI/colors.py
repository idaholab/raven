from matplotlib import cm, colors
import numpy as np
import itertools
from PySide.QtGui import QColor

minPenColor = QColor(33,102,172) #QColor(57,94,150)
minBrushColor = QColor(67,147,195) #QColor(114,143,184)
inactiveMinPenColor = minBrushColor.lighter() #minPenColor.lighter()
inactiveMinBrushColor = minBrushColor.lighter()

maxPenColor = QColor(178,24,43) #QColor(211,156,95)
maxBrushColor = QColor(214,96,77) #QColor(251,184,108)
inactiveMaxPenColor = maxBrushColor.lighter() # maxPenColor.lighter()
inactiveMaxBrushColor = maxBrushColor.lighter()


TolColors = ['#88CCEE', '#DDCC77', '#AA4499', '#117733', '#332288', '#999933',
             '#44AA99', '#882255', '#CC6677']

colorCycle = itertools.cycle(TolColors)

# *****************************************************************************
# Helper Code
# *****************************************************************************
import math

def div(nom, denom):
  if denom <= 0.00001:
    return nom
  else:
    return nom / denom


def divInt(nom, denom):
  if denom <= 0.00001:
    return nom
  else:
    return nom // denom


def mod(nom, denom):
  if denom <= 0.00001:
    return nom
  else:
    return nom % denom

def deleteListItem(curList, index):
  if not curList:
    return
  del curList[index % len(curList)]


def setListIndexTo(curList, index, value):
  if not curList:
    return
  curList[index % len(curList)] = value


def getIndexBoolList(curList, index):
  if not curList:
    return bool()
  return curList[index % len(curList)]


def getIndexFloatList(curList, index):
  if not curList:
    return float()
  return curList[index % len(curList)]


def getIndexIntList(curList, index):
  if not curList:
    return int()
  return curList[index % len(curList)]


def getIndexStringList(curList, index):
  if not curList:
    return str()
  return curList[index % len(curList)]

def getCharFromString(curString, index):
  if not curString:
    return ''
  return curString[index % len(curString)]


def saveChr(number):
    return chr(number % 128)


def saveOrd(literal):
    if len(literal) <= 0:
        return 32
    return ord(literal[0])


def saveSplit(curString, separator):
  if not separator:
    return []
  return curString.split(separator)

def saveRange(start, end):
  if end > start and abs(start - end) > 10000:
    return range(start, start + 10000)
  return range(start, end)

# *****************************************************************************

#  evolved function
def evolve(in0, in1, in2, in3):
  # stop.value is a boolean flag which should be used to check if the EA wants the program to stop.value
  <insertCodeHere>
  return res0


def fitnessTrainingCase(i, o):
  eval = evolve(i[0], i[1], i[2], i[3])

  return [0 if eval == o[0] else 1]


#  function to evaluate fitness
def fitness():
  error = []
  cases = []
  for (i, o) in zip(inval, outval):
    values = fitnessTrainingCase(i, o)
    error.extend(values)
    cases.append(all(v < 0.000000001 for v in values))

  return error, cases


import sys
quality = sys.maxsize
caseQuality, cases = fitness()
quality = sum(caseQuality)

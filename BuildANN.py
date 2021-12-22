'''
BuildANN.py : Builds the Artifical Neural Network and stores the model, which predict.py can use
'''

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle

dataFrame = pd.read_csv("CompleteDataOriginalMod.csv",header = 0)

allCircuits = set(dataFrame["Circuits"])

f = open("Values.txt","w")
for i in allCircuits:
	f.write(str(i) + ",")
f.write("\n\n\n")
# Make a list of all circuits used in the data, and circuits used for each release

	
'''
Dummification : Process of converting categorical attributes into numerical formats so that statistical models can analyze them
Eg : 
If a record can have categorical values "A", "B", and "C" (which have no numeric significance), three additional columns are added which conceptually mean "isA", "isB", and "isC"
The record which has value "A", would have "1", "0", "0" in the corresponding columns.

Here, our categorical attributes are Industry, Circuits, Country, and Client
'''




# Perform in-built dummification for other attributes
	
dummifiedCountry = pd.get_dummies(dataFrame['Country'])
dummifiedIndustry = pd.get_dummies(dataFrame['Industry'])
dummifiedClient = pd.get_dummies(dataFrame['Client'])
dummifiedCircuit = pd.get_dummies(dataFrame['Circuits'])

# Merge dummified attributes back into original dataframe
for colName in dummifiedCountry:
	dataFrame["Country_" + str(colName)] = dummifiedCountry[colName]
for colName in dummifiedIndustry:
	dataFrame["Industry_" + str(colName)] = dummifiedIndustry[colName]
for colName in dummifiedClient:
	dataFrame["Client_"+str(colName)] = dummifiedClient[colName]
for colName in dummifiedCircuit:
	dataFrame["Circuit_"+str(colName)] = dummifiedCircuit[colName]



# Store ID Mappings which predict.py uses, for country, client, and organization
allIdMappings = dict()

# Country
hardCodedCountryMapping = {"USA" : 233, "France" : 74, "Canada" : 39, "Australia" : 14}	#DB inconsistency; Unsure of actual mappings
allIdMappings.update(hardCodedCountryMapping)

# Industry

allIndustries = set(dataFrame["Industry"])
allIndustries = set(map(str,allIndustries))
industryMappings = dict()
industryFile = open("Industry.csv")
industryFile.readline()

for line in industryFile:
	line = line.strip().split(",")
	if line[0] in allIndustries and line[2] not in industryMappings:
		industryMappings[line[2]] = line[0]
allIdMappings.update(industryMappings)


# Organization

allOrganizations = set(dataFrame["Client"])
allOrganizations = set(map(str,allOrganizations))
organizationMappings = dict()
organizationFile = open("Organization.csv")
organizationFile.readline()

for line in organizationFile:
	line = line.strip().split(",")
	if line[0] in allOrganizations:
		
		organizationMappings[line[1]] = line[0]


for org in organizationMappings:
	f.write("\"" + org + "\",")
f.write("\n\n\n")	
		
allIdMappings.update(organizationMappings)
	
# Remove original attributes as dummified version has been added to the dataframe 	
del dataFrame['Country']
del dataFrame['Industry']
del dataFrame['Client']
del dataFrame['Circuits']

# Include rounded MaxCount 
dataFrame['Count'] = list(map(lambda count : round(count/100),dataFrame['Count']))

# Split dataframe into dependent and independent attributes in order to build and train model
X = dataFrame[dataFrame.columns[1:]]
y = dataFrame[dataFrame.columns[0]]
annModel = MLPClassifier(hidden_layer_sizes = (13,13,13), max_iter = 500)
annModel.fit(X,y)




		
# Store binary dumps so as to not build the model every time
pickle.dump(annModel,open("ANNModel.pkl","wb") )
pickle.dump(list(dataFrame.columns.values),open("ColumnNames.pkl","wb")) 
pickle.dump(allIdMappings,open("AllIdMappings.pkl","wb"))
'''
predict.py : Uses the model object created by BuildANN.py and provides a prediction in JSON. 
'''

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import pickle

app = Flask(__name__)
CORS(app)

# To consume this API over command-line: 
# curl -X POST -H "Content-Type: application/json" -d "{ \"key1\": \"value1\" }" http://localhost:3000/api/method
# Input JSON are parameters like Country, Organization, Industry, and Circuits


# This function converts the parameters for which prediction is to be made, into the dummified version.
def getDummiedTest(input):

	#Defaults in case user does not specify all features
	defaultValues = {
		"Country" : 0, 
		"Industry" : 0,
		"Client" : 0,
		"Circuit" : [],
		"allTimes" : [0.0,0.041,0.083, 0.125, 0.166, 0.208, 0.250, 0.291, 0.333,0.375, 0.416, 0.458, 0.500, 0.541, 0.583, 0.625, 0.666, 0.708, 0.750 , 0.791, 0.833, 0.875, 0.916, 0.958]	
	}
	
	defaultValues.update(input)
	
	global colNames
	masterTestList = []
	# Add the 1s and 0s appropriately as required for dummification

	testList = [0] * (len(colNames) - 1)
	allTestLists = []
	if defaultValues["Country"]:
		testList[colNames.index("Country_" + str(defaultValues["Country"])) - 1] = 1
	if defaultValues["Industry"]:
		testList[colNames.index("Industry_" + str(defaultValues["Industry"])) - 1] = 1
	if defaultValues["Client"]:
		testList[colNames.index("Client_" + str(defaultValues["Client"])) - 1] = 1
	for c in defaultValues["Circuit"]:
		testList[colNames.index("Circuit_" + str(c)) - 1] = 1
	
	#allTimes is a list of epoch times, for which the predictions are made 
		for i in defaultValues["allTimes"]:
			testList[0] = i
			allTestLists.append(testList[:])
		masterTestList.append(allTestLists[:])

	print("MASTER : ",len(masterTestList))
	print("MASTER[0] : ",len(masterTestList[0]))	
	return masterTestList



#This function maps strings like country name / organization name to their Database IDs
def mapInputToId(values):
	global allIdMappings
	for inputParameter in values:
		if not inputParameter == "Circuit":
			values[inputParameter] = allIdMappings[values[inputParameter]]
	return values

	
#Exposed API which expects json as part of POST request; Returns predictions as JSON
@app.route('/predict', methods=['POST','GET'])

def predict():
	input = request.get_json()
	print(input)
	mappedValues = mapInputToId(input)

	dummifiedLists = getDummiedTest(mappedValues)

	peakViewsSums = [0] * 24
	for circuitWiseList in dummifiedLists:
		peakViews = annModel.predict(circuitWiseList)
		for i in range(24):
			peakViewsSums[i] += peakViews[i]
	
	
	prediction = dict()
		
	for i in range(len(peakViewsSums)):
		prediction[i] = peakViewsSums[i]*100

	return jsonify(prediction)

if __name__ == '__main__':
	annModel = pickle.load(open('ANNmodel.pkl',"rb"))
	colNames = pickle.load(open("ColumnNames.pkl", "rb"))
	allIdMappings = pickle.load(open("AllIdMappings.pkl", "rb"))
	app.run(port=5000)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from django.shortcuts import render

# Create your views here.
def detection(request):

	columns = ['age', 'sex', 'fever', 'cough', 'fatigue', 'abdominal pain', 'diarrhea',
       'malaise', 'pneumonia', 'aching muscles', 'anorexia', 'asymptomatic',
       'chest discomfort', 'dyspnea', 'nausea', 'vomitting', 'chills',
       'conjuctivitis', 'joint pain', 'headache', 'weakness', 'sore throat',
       'sneezing', 'rhinorrhea', 'dizziness', 'runny nose',
       'difficulty walking', 'sputum', 'pneumonitis', 'physical discomfort',
       'toothache', 'wheezing', 'dry mouth', 'sweating']

	sentences = ""
	op = ""

	if request.method == "POST":

		# Taking all the feature inputs
		age = request.POST['age']
		sex = request.POST['sex']
		fever = request.POST['fever']
		cough = request.POST['cough']
		fatigue = request.POST['fatigue']
		abdominalPain = request.POST['abdominalPain']
		diarrhea = request.POST['diarrhea']
		malaise = request.POST['malaise']
		pneumonia = request.POST['pneumonia']
		achingMuscles = request.POST['achingMuscles']
		anorexia = request.POST['anorexia']
		asymptomatic = request.POST['asymptomatic']
		chestDiscomfort = request.POST['chestDiscomfort']
		dyspnea = request.POST['dyspnea']
		nausea = request.POST['nausea']
		vomitting = request.POST['vomitting']
		chills = request.POST['chills']
		conjuctivitis = request.POST['conjuctivitis']
		jointPain = request.POST['jointPain']
		headache = request.POST['headache']
		weakness = request.POST['weakness']
		soreThroat = request.POST['soreThroat']
		sneezing = request.POST['sneezing']
		rhinorrhea = request.POST['rhinorrhea']
		dizziness = request.POST['dizziness']
		runnyNose = request.POST['runnyNose']
		difficultyWalking = request.POST['difficultyWalking']
		sputum = request.POST['sputum']
		pneumonitis = request.POST['pneumonitis']
		physicalDiscomfort = request.POST['physicalDiscomfort']
		toothache = request.POST['toothache']
		wheezing = request.POST['wheezing']
		dryMouth = request.POST['dryMouth']
		sweating = request.POST['sweating']


		# Initializing the DataFrame
		inpDF = pd.DataFrame(np.nan, index=[0], columns=columns)
		# inp = input("Input the text you want to verify: ")

		# Initializng all the feature columns with 0 values
		inpDF["age"] = np.zeros((1,1),dtype=float)
		inpDF["sex"] = np.zeros((1,1),dtype=float)
		inpDF["fever"] = np.zeros((1,1),dtype=float)
		inpDF["cough"] = np.zeros((1,1),dtype=float)
		inpDF["fatigue"] = np.zeros((1,1),dtype=float)
		inpDF["abdominal pain"] = np.zeros((1,1),dtype=float)
		inpDF["diarrhea"] = np.zeros((1,1),dtype=float)
		inpDF["malaise"] = np.zeros((1,1),dtype=float)
		inpDF["pneumonia"] = np.zeros((1,1),dtype=float)
		inpDF["aching muscles"] = np.zeros((1,1),dtype=float)
		inpDF["anorexia"] = np.zeros((1,1),dtype=float)
		inpDF["asymptomatic"] = np.zeros((1,1),dtype=float)
		inpDF["chest discomfort"] = np.zeros((1,1),dtype=float)
		inpDF["dyspnea"] = np.zeros((1,1),dtype=float)
		inpDF["nausea"] = np.zeros((1,1),dtype=float)
		inpDF["vomitting"] = np.zeros((1,1),dtype=float)
		inpDF["chills"] = np.zeros((1,1),dtype=float)
		inpDF["conjuctivitis"] = np.zeros((1,1),dtype=float)
		inpDF["joint pain"] = np.zeros((1,1),dtype=float)
		inpDF["headache"] = np.zeros((1,1),dtype=float)
		inpDF["weakness"] = np.zeros((1,1),dtype=float)
		inpDF["sore throat"] = np.zeros((1,1),dtype=float)
		inpDF["sneezing"] = np.zeros((1,1),dtype=float)
		inpDF["rhinorrhea"] = np.zeros((1,1),dtype=float)
		inpDF["dizziness"] = np.zeros((1,1),dtype=float)
		inpDF["runny nose"] = np.zeros((1,1),dtype=float)
		inpDF["difficulty walking"] = np.zeros((1,1),dtype=float)
		inpDF["sputum"] = np.zeros((1,1),dtype=float)
		inpDF["pneumonitis"] = np.zeros((1,1),dtype=float)
		inpDF["physical discomfort"] = np.zeros((1,1),dtype=float)
		inpDF["toothache"] = np.zeros((1,1),dtype=float)
		inpDF["wheezing"] = np.zeros((1,1),dtype=float)
		inpDF["dry mouth"] = np.zeros((1,1),dtype=float)
		inpDF["sweating"] = np.zeros((1,1),dtype=float)



		# Now, the 0 values on feature table is replace by the user inputs
		inpDF["age"] = age
		inpDF["sex"] = sex
		inpDF["fever"] = fever
		inpDF["cough"] = cough
		inpDF["fatigue"] = fatigue
		inpDF["abdominal pain"] = abdominalPain
		inpDF["diarrhea"] = diarrhea
		inpDF["malaise"] = malaise
		inpDF["pneumonia"] = pneumonia
		inpDF["aching muscles"] = achingMuscles
		inpDF["anorexia"] = anorexia
		inpDF["asymptomatic"] = asymptomatic
		inpDF["chest discomfort"] = chestDiscomfort
		inpDF["dyspnea"] = dyspnea
		inpDF["nausea"] = nausea
		inpDF["vomitting"] = vomitting
		inpDF["chills"] = chills
		inpDF["conjuctivitis"] = conjuctivitis
		inpDF["joint pain"] = jointPain
		inpDF["headache"] = headache
		inpDF["weakness"] = weakness
		inpDF["sore throat"] = soreThroat
		inpDF["sneezing"] = sneezing
		inpDF["rhinorrhea"] = rhinorrhea
		inpDF["dizziness"] = dizziness
		inpDF["runny nose"] = runnyNose
		inpDF["difficulty walking"] = difficultyWalking
		inpDF["sputum"] = sputum
		inpDF["pneumonitis"] = pneumonitis
		inpDF["physical discomfort"] = physicalDiscomfort
		inpDF["toothache"] = toothache
		inpDF["wheezing"] = wheezing
		inpDF["dry mouth"] = dryMouth
		inpDF["sweating"] = sweating


		print(inpDF)

		X = inpDF.copy()

		featureModel = 'Covid/Models/svmModel.sav'
		loadedFEATmodel = pickle.load(open(featureModel, 'rb'))
		featresult = loadedFEATmodel.predict(X)


		op = (featresult[0])

		print("--------")
		print(op)
		print("--------")

	return render(request, "Covid/detector.html", context= {'text' : op})

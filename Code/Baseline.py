##
## name: Baseline.py
## purpose: Baseline implementation
## date: 04/20/2019
##
##

import Sandbox
import numpy as np
import matplotlib.pyplot as plt
from time import clock

if __name__ == '__main__':

	## Parameters
	outputDir = "../Data/Sumedh"
	length = 32
	width = 32
	hiddenLayers = [[7, 7]]
	etas = [1.1]
	trainingLen = 24
	validationLen = 8
	testingLen = 8
	tvt = [[22, 10, 10]]
	maxEpochs = [10]
	errorThreshold = 0.05
	thresholds = [0.5]
	iterations = range(50)

	## Metrics
	confusionMatrices = []
	computingTimes = []

	## Start training time 
	start_time = clock()
	for t in tvt:
		for iteration in iterations:
			for hl in hiddenLayers:
				for epochs in maxEpochs:
					for eta in etas:
						for threshold in thresholds:
							baseline = Sandbox.Sandbox()

							csv = "../Data/Sumedh/DL_liver_32_32.csv"
							files = [
								"../Data/Raghav/Images_png/002648_01_01_061.png",
								"../Data/Raghav/Images_png/002648_02_01_311.png",
								"../Data/Raghav/Images_png/002677_01_01_044.png",
								"../Data/Raghav/Images_png/002680_04_01_050.png",
								"../Data/Raghav/Images_png/002680_05_01_305.png",
								"../Data/Raghav/Images_png/002680_06_01_313.png",
								"../Data/Raghav/Images_png/002680_06_01_323.png",
								"../Data/Raghav/Images_png/002680_06_01_344.png",
								"../Data/Raghav/Images_png/003599_01_01_247.png",
								"../Data/Raghav/Images_png/003599_01_01_260.png",
								"../Data/Raghav/Images_png/003599_01_01_270.png",
								"../Data/Raghav/Images_png/003620_01_02_280.png",
								"../Data/Raghav/Images_png/003657_01_01_049.png",
								"../Data/Raghav/Images_png/003657_01_01_050.png",
								"../Data/Raghav/Images_png/003657_01_01_069.png",
								"../Data/Raghav/Images_png/003663_01_03_190.png",
								"../Data/Raghav/Images_png/003663_01_03_209.png",
								"../Data/Raghav/Images_png/003663_01_03_237.png",
								"../Data/Raghav/Images_png/003663_01_03_304.png",
								"../Data/Raghav/Images_png/003663_01_03_308.png",
								"../Data/Raghav/Images_png/003663_02_02_276.png",
								"../Data/Raghav/Images_png/003667_01_01_052.png",
								"../Data/Raghav/Images_png/004416_01_01_047.png",
								"../Data/Sumedh/Images_png/000016_02_01/253.png",
								"../Data/Sumedh/Images_png/000016_02_01/253.png",
								"../Data/Sumedh/Images_png/000047_10_01/051.png",
								"../Data/Sumedh/Images_png/000047_10_01/054.png",
								"../Data/Sumedh/Images_png/000053_03_01/044.png",
								"../Data/Sumedh/Images_png/000053_04_01/042.png",
								"../Data/Sumedh/Images_png/000053_05_01/044.png",
								"../Data/Sumedh/Images_png/000053_06_01/222.png",
								"../Data/Sumedh/Images_png/000053_06_01/297.png",
								"../Data/Sumedh/Images_png/000069_02_01/058.png",
								"../Data/Sumedh/Images_png/000069_03_01/058.png",
								"../Data/Sumedh/Images_png/000078_01_01/124.png",
								"../Data/Sumedh/Images_png/000078_02_01/063.png",
								"../Data/Sumedh/Images_png/000088_02_01/306.png",
								"../Data/Sumedh/Images_png/000120_02_01/061.png",
								"../Data/Sumedh/Images_png/000122_01_01/063.png",
								"../Data/Sumedh/Images_png/000122_02_01/062.png",
								"../Data/Sumedh/Images_png/000126_02_01/044.png",
								"../Data/Sumedh/Images_png/000126_02_01/047.png",
								"../Data/Sumedh/Images_png/000268_03_01/058.png",
								"../Data/Sumedh/Images_png/000286_03_01/265.png",
								"../Data/Sumedh/Images_png/000286_04_01/052.png",
								"../Data/Sumedh/Images_png/000286_05_01/049.png",
								"../Data/Sumedh/Images_png/000286_05_01/052.png",
								"../Data/Sumedh/Images_png/000286_05_01/116.png"

							]
							links = [
							    'https://nihcc.box.com/shared/static/sp5y2k799v4x1x77f7w1aqp26uyfq7qz.zip',
							    'https://nihcc.box.com/shared/static/l9e1ys5e48qq8s409ua3uv6uwuko0y5c.zip',
							    'https://nihcc.box.com/shared/static/48jotosvbrw0rlke4u88tzadmabcp72r.zip',
							    'https://nihcc.box.com/shared/static/xa3rjr6nzej6yfgzj9z6hf97ljpq1wkm.zip',
							    'https://nihcc.box.com/shared/static/58ix4lxaadjxvjzq4am5ehpzhdvzl7os.zip',
							    'https://nihcc.box.com/shared/static/cfouy1al16n0linxqt504n3macomhdj8.zip',
							    'https://nihcc.box.com/shared/static/z84jjstqfrhhlr7jikwsvcdutl7jnk78.zip',
							    'https://nihcc.box.com/shared/static/6viu9bqirhjjz34xhd1nttcqurez8654.zip',
							    'https://nihcc.box.com/shared/static/9ii2xb6z7869khz9xxrwcx1393a05610.zip',
							    'https://nihcc.box.com/shared/static/2c7y53eees3a3vdls5preayjaf0mc3bn.zip',

							    'https://nihcc.box.com/shared/static/2zsqpzru46wsp0f99eaag5yiad42iezz.zip',
							    'https://nihcc.box.com/shared/static/8v8kfhgyngceiu6cr4sq1o8yftu8162m.zip',
							    'https://nihcc.box.com/shared/static/jl8ic5cq84e1ijy6z8h52mhnzfqj36q6.zip',
							    'https://nihcc.box.com/shared/static/un990ghdh14hp0k7zm8m4qkqrbc0qfu5.zip',
							    'https://nihcc.box.com/shared/static/kxvbvri827o1ssl7l4ji1fngfe0pbt4p.zip',
							    'https://nihcc.box.com/shared/static/h1jhw1bee3c08pgk537j02q6ue2brxmb.zip',
							    'https://nihcc.box.com/shared/static/78hamrdfzjzevrxqfr95h1jqzdqndi19.zip',
							    'https://nihcc.box.com/shared/static/kca6qlkgejyxtsgjgvyoku3z745wbgkc.zip',
							    'https://nihcc.box.com/shared/static/e8yrtq31g0d8yhjrl6kjplffbsxoc5aw.zip',
							    'https://nihcc.box.com/shared/static/vomu8feie1qembrsfy2yaq36cimvymj8.zip',

							    'https://nihcc.box.com/shared/static/ecwyyx47p2jd621wt5c5tc92dselz9nx.zip',
							    'https://nihcc.box.com/shared/static/fbnafa8rj00y0b5tq05wld0vbgvxnbpe.zip',
							    'https://nihcc.box.com/shared/static/50v75duviqrhaj1h7a1v3gm6iv9d58en.zip',
							    'https://nihcc.box.com/shared/static/oylbi4bmcnr2o65id2v9rfnqp16l3hp0.zip',
							    'https://nihcc.box.com/shared/static/mw15sn09vriv3f1lrlnh3plz7pxt4hoo.zip',
							    'https://nihcc.box.com/shared/static/zi68hd5o6dajgimnw5fiu7sh63kah5sd.zip',
							    'https://nihcc.box.com/shared/static/3yiszde3vlklv4xoj1m7k0syqo3yy5ec.zip',
							    'https://nihcc.box.com/shared/static/w2v86eshepbix9u3813m70d8zqe735xq.zip',
							    'https://nihcc.box.com/shared/static/0cf5w11yvecfq34sd09qol5atzk1a4ql.zip',
							    'https://nihcc.box.com/shared/static/275en88yybbvzf7hhsbl6d7kghfxfshi.zip',

							    'https://nihcc.box.com/shared/static/l52tpmmkgjlfa065ow8czhivhu5vx27n.zip',
							    'https://nihcc.box.com/shared/static/p89awvi7nj0yov1l2o9hzi5l3q183lqe.zip',
							    'https://nihcc.box.com/shared/static/or9m7tqbrayvtuppsm4epwsl9rog94o8.zip',
							    'https://nihcc.box.com/shared/static/vuac680472w3r7i859b0ng7fcxf71wev.zip',
							    'https://nihcc.box.com/shared/static/pllix2czjvoykgbd8syzq9gq5wkofps6.zip',
							    'https://nihcc.box.com/shared/static/2dn2kipkkya5zuusll4jlyil3cqzboyk.zip',
							    'https://nihcc.box.com/shared/static/peva7rpx9lww6zgpd0n8olpo3b2n05ft.zip',
							    'https://nihcc.box.com/shared/static/2fda8akx3r3mhkts4v6mg3si7dipr7rg.zip',
							    'https://nihcc.box.com/shared/static/ijd3kwljgpgynfwj0vhj5j5aurzjpwxp.zip',
							    'https://nihcc.box.com/shared/static/nc6rwjixplkc5cx983mng9mwe99j8oa2.zip',

							    'https://nihcc.box.com/shared/static/rhnfkwctdcb6y92gn7u98pept6qjfaud.zip',
							    'https://nihcc.box.com/shared/static/7315e79xqm72osa4869oqkb2o0wayz6k.zip',
							    'https://nihcc.box.com/shared/static/4nbwf4j9ejhm2ozv8mz3x9jcji6knhhk.zip',
							    'https://nihcc.box.com/shared/static/1lhhx2uc7w14bt70de0bzcja199k62vn.zip',
							    'https://nihcc.box.com/shared/static/guho09wmfnlpmg64npz78m4jg5oxqnbo.zip',
							    'https://nihcc.box.com/shared/static/epu016ga5dh01s9ynlbioyjbi2dua02x.zip',
							    'https://nihcc.box.com/shared/static/b4ebv95vpr55jqghf6bthg92vktocdkg.zip',
							    'https://nihcc.box.com/shared/static/byl9pk2y727wpvk0pju4ls4oomz9du6t.zip',
							    'https://nihcc.box.com/shared/static/kisfbpualo24dhby243nuyfr8bszkqg1.zip',
							    'https://nihcc.box.com/shared/static/rs1s5ouk4l3icu1n6vyf63r2uhmnv6wz.zip',

							    'https://nihcc.box.com/shared/static/7tvrneuqt4eq4q1d7lj0fnafn15hu9oj.zip',
							    'https://nihcc.box.com/shared/static/gjo530t0dgeci3hizcfdvubr2n3mzmtu.zip',
							    'https://nihcc.box.com/shared/static/7x4pvrdu0lhazj83sdee7nr0zj0s1t0v.zip',
							    'https://nihcc.box.com/shared/static/z7s2zzdtxe696rlo16cqf5pxahpl8dup.zip',
							    'https://nihcc.box.com/shared/static/shr998yp51gf2y5jj7jqxz2ht8lcbril.zip',
							    'https://nihcc.box.com/shared/static/kqg4peb9j53ljhrxe3l3zrj4ac6xogif.zip'
							]

							## Initialize DL data tank
							baseline.initDLDataTank(csv,
								files, 
								[], outputDir,
								length, width)



												
							## Initialize FFBP
							baseline.initFFBP(hl,
								eta)

							## Run Train-Validation-Testing scheme
							baseline.setLesionThreshold(threshold)
							baseline.runExperiment(t[0],
								t[1],
								t[2],
								epochs,
								errorThreshold)
							## Confusion matrix
							confusionMatrix = baseline.getConfusionMatrix()
							print(confusionMatrix)
							confusionMatrices.append(confusionMatrix)
							## Computing times
							computingTime = baseline.getComputingTime()
							print(computingTime)
							computingTimes.append(computingTime)
	## End training time
	end_time = clock()

	## Get confusion matrix average
	'''
	average_error = 0.0
	average_accuracy = 0.0
	average_sensitivity = 0.0
	average_false_positives = 0.0
	for cm in confusionMatrices:
		average_error += (cm[0][1]+cm[1][0])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
		average_accuracy += (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
		average_sensitivity += (cm[1][1]/(cm[1][1]+cm[1][0]))
		average_false_positives += cm[0][1]
	'''
	total_cm =[[0.0, 0.0], 
			[0.0, 0.0]]
	for cm in confusionMatrices:
		total_cm[0][0] += float(cm[0][0])
		total_cm[0][1] += float(cm[0][1])
		total_cm[1][0] += float(cm[1][0])
		total_cm[1][1] += float(cm[1][1])

	print([(total_cm[0][0]+total_cm[0][1])/(total_cm[0][0]+total_cm[0][1]+total_cm[1][0]+total_cm[1][1]),
		(total_cm[1][1]+total_cm[1][0])/(total_cm[0][0]+total_cm[0][1]+total_cm[1][0]+total_cm[1][1]),
		(total_cm[1][1])/(total_cm[1][1]+total_cm[1][0]),
		(total_cm[0][1])/len(iterations),
		end_time-start_time])

	'''
	## Plot sensitivity curve
	true_positive_rate = []
	false_positive_rate = []
	for cf in confusionMatrices:
		print(cf)
		tpr = 0.0
		fpr = 0.0
		if float(cf[1][0]+cf[1][1]) != 0.0:
			tpr = float(cf[1][1])/float(cf[1][0]+cf[1][1])
		else:
			tpr = 0.0
		print("True positive rate: "+str(tpr))
		if float(cf[0][1]+cf[0][0]) != 0.0:
			fpr = float(cf[0][1])/float(cf[0][1]+cf[0][0])
		else:
			fpr = 1.0
		print("False positive rate: "+str(fpr))
		true_positive_rate.append(tpr)
		false_positive_rate.append(fpr)
	plt.scatter(false_positive_rate, true_positive_rate)
	for i,txt in enumerate(iterations):
		plt.annotate(txt, 
			(false_positive_rate[i], 
			true_positive_rate[i]))

	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.title("ROC Curve")
	plt.show()

	## Plot computing time box-plots
	plt.boxplot(computingTimes)# , labels=iterations)
	plt.xlabel("Iterations")
	plt.ylabel("Times (s)")
	plt.title("Computing Times")
	plt.show()
	'''


	
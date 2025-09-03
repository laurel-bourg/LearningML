

def slope(w, b, x): # solves for the value along the predicted slope
	return b + w * x
	

def gradientDesc(feature, label): # main function using gradient descent to find minimum loss
	weight = 0
	bias = 0
	n = len(feature)
	cont = True
	prevLoss = 0 # stores the previous loss for later comparison
	iter = 0
	
	while cont:
		lossTot  = 0
		for i in range(0, len(feature)):
			y = slope(weight, bias, feature[i])
			lossTot += (label[i] - y)**2
	
		loss = lossTot/n
		# print("Loss: " + str(loss))
		if abs(loss - prevLoss) < 0.005:
			cont = False
		
		weightSlope = 0
		for i in range(0, len(feature)): # find weight derivative
			weightSlope += (slope(weight, bias, feature[i]) - label[i]) * 2 * feature[i]
		weightSlope = weightSlope/n

		biasSlope = 0
		for i in range(0, len(feature)): # find bias derivative
			biasSlope += (slope(weight, bias, feature[i]) - label[i]) * 2
		biasSlope = biasSlope/n

		weight = weight - (0.01 * weightSlope)
		# print("Weight: " + str(weight))
		bias = bias - (0.01 * biasSlope)
		# print("Bias: " + str(bias))
		prevLoss = loss
		iter += 1
	
	print("Final Loss value: " + str(loss))
	print("Iterations: " + str(iter))
	
	return loss

# Example Case

pounds = [3.5, 3.69, 3.44, 3.43, 4.34, 4.42, 2.37]

miles = [18, 15, 18, 16, 15, 14, 24]

gradientDesc(pounds, miles)


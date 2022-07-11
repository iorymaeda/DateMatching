def findDecision(obj): #obj[0]: distance
	# {"feature": "distance", "instances": 66, "metric_value": 0.8454, "depth": 1}
	if obj[0]<=0.612715352403554:
		return 'No'
	elif obj[0]>0.612715352403554:
		return 'Yes'
	else: return 'Yes'

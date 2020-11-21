class Generator : 

	def generate(input): 
		output = [input] * 5 
		for i in range (0, 5): 
			output[i] += chr(i + 97) * i + '<br>'
		 
		merged = ''
		merged = merged.join(output)
		merged = ''.join(merged.split())
		return merged

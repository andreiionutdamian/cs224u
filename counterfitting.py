
def normalise_word_vectors(word_vectors, norm=1.0):
    """
    This method normalises the collection of word vectors provided in the word_vectors dictionary.
    
    
    https://raw.githubusercontent.com/nmrksic/counter-fitting/master/counterfitting.py
    """
    for word in word_vectors:
        word_vectors[word] /= np.sqrt((word_vectors[word]**2).sum() + 1e-6)
        word_vectors[word] = word_vectors[word] * norm
    return word_vectors

def distance(v1, v2, normalised_vectors=True):
	"""
	Returns the cosine distance between two vectors. 
	If the vectors are normalised, there is no need for the denominator, which is always one. 

    https://raw.githubusercontent.com/nmrksic/counter-fitting/master/counterfitting.py
	"""
	if normalised_vectors:
		return 1 - np.dot(v1, v2)
	else:
		return 1 - np.dot(v1, v2) / ( np.linalg.norm(v1) * np.linalg.norm(v2) )


def compute_vsp_pairs(dct_word_vectors, rho=0.2):
    """
    This method returns a dictionary with all word pairs which are closer together than rho.
    Each pair maps to the original distance in the vector space. 
    In order to manage memory, this method computes dot-products of different subsets of word 
    vectors and then reconstructs the indices of the word vectors that are deemed to be similar.
    
    https://raw.githubusercontent.com/nmrksic/counter-fitting/master/counterfitting.py
    """
    print("Pre-computing word pairs relevant for Vector Space Preservation (VSP). Rho =", rho, flush=True)
	
    word_vectors = dct_word_vectors
    vsp_pairs = {}

    threshold = 1 - rho 
    vocabulary = list(dct_word_vectors.keys())
    num_words = len(vocabulary)

    step_size = 1000 # Number of word vectors to consider at each iteration. 
    vector_size = word_vectors[list(word_vectors.keys())[0]].shape[0]

	# ranges of word vector indices to consider:
    list_of_ranges = []

    left_range_limit = 0
    while left_range_limit < num_words:
      curr_range = (left_range_limit, min(num_words, left_range_limit + step_size))
      list_of_ranges.append(curr_range)
      left_range_limit  += step_size

    range_count = len(list_of_ranges)

	# now compute similarities between words in each word range:
    for left_range in range(range_count):
      for right_range in range(left_range, range_count):

			# offsets of the current word ranges:
            left_translation = list_of_ranges[left_range][0]
            right_translation = list_of_ranges[right_range][0]

			# copy the word vectors of the current word ranges:
            vectors_left = np.zeros((step_size, vector_size), dtype="float32")
            vectors_right = np.zeros((step_size, vector_size), dtype="float32")

			# two iterations as the two ranges need not be same length (implicit zero-padding):
            full_left_range = range(list_of_ranges[left_range][0], list_of_ranges[left_range][1])		
            full_right_range = range(list_of_ranges[right_range][0], list_of_ranges[right_range][1])
			
            for iter_idx in full_left_range:
              vectors_left[iter_idx - left_translation, :] = word_vectors[vocabulary[iter_idx]]

            for iter_idx in full_right_range:
              vectors_right[iter_idx - right_translation, :] = word_vectors[vocabulary[iter_idx]]

			# now compute the correlations between the two sets of word vectors: 
            dot_product = vectors_left.dot(vectors_right.T)

			# find the indices of those word pairs whose dot product is above the threshold:
            indices = np.where(dot_product >= threshold)
            
            num_pairs = indices[0].shape[0]
            left_indices = indices[0]
            right_indices = indices[1]
            
            
            for iter_idx in range(0, num_pairs):
              left_word = vocabulary[left_translation + left_indices[iter_idx]]
              right_word = vocabulary[right_translation + right_indices[iter_idx]]
              
              if left_word != right_word:
				# reconstruct the cosine distance and add word pair (both permutations):
                score = 1 - dot_product[left_indices[iter_idx], right_indices[iter_idx]]
                vsp_pairs[(left_word, right_word)] = score
                vsp_pairs[(right_word, left_word)] = score
		
    return vsp_pairs



def vector_partial_gradient(u, v, normalised_vectors=True):
	"""
	This function returns the gradient of cosine distance: \frac{ \partial dist(u,v)}{ \partial u}
	If they are both of norm 1 (we do full batch and we renormalise at every step), we can save some time.
    
    https://raw.githubusercontent.com/nmrksic/counter-fitting/master/counterfitting.py
	"""

	if normalised_vectors:
		gradient = u * np.dot(u,v)  - v 
	else:		
		norm_u = np.linalg.norm(u)
		norm_v = np.linalg.norm(v)
		nominator = u * np.dot(u,v) - v * np.power(norm_u, 2)
		denominator = norm_v * np.power(norm_u, 3)
		gradient = nominator / denominator

	return gradient


def one_step_SGD(word_vectors, synonym_pairs, antonym_pairs, vsp_pairs, 
                 delta=1.0, hyper_k1=0.1, hyper_k2=0.1, hyper_k3=0.1, gamma=0):
    """
    This method performs a step of SGD to optimise the counterfitting cost function.

    https://raw.githubusercontent.com/nmrksic/counter-fitting/master/counterfitting.py
    """
    from copy import deepcopy
    new_word_vectors = deepcopy(word_vectors)

    gradient_updates = {}
    update_count = {}
    oa_updates = {}
    vsp_updates = {}

    # AR term:
    if hyper_k1 > 0.0:
      for i, (word_i, word_j) in enumerate(antonym_pairs):
          print("\r    Processing antonym pairs {:.1f}%...".format(i/len(antonym_pairs)*100), flush=True, end='')
  
          current_distance = distance(new_word_vectors[word_i], new_word_vectors[word_j])
  
          if current_distance < delta:
      
              gradient = vector_partial_gradient( new_word_vectors[word_i], new_word_vectors[word_j])
              gradient = gradient * hyper_k1 
  
              if word_i in gradient_updates:
                  gradient_updates[word_i] += gradient
                  update_count[word_i] += 1
              else:
                  gradient_updates[word_i] = gradient
                  update_count[word_i] = 1

    # SA term:
    if hyper_k2 > 0.0:
      for i, (word_i, word_j) in enumerate(synonym_pairs):
          print("\r    Processing synonym pairs {:.1f}%...".format(i/len(synonym_pairs)*100), flush=True, end='')
      
          current_distance = distance(new_word_vectors[word_i], new_word_vectors[word_j])
  
          if current_distance > gamma: 
          
              gradient = vector_partial_gradient(new_word_vectors[word_j], new_word_vectors[word_i])
              gradient = gradient * hyper_k2 
  
              if word_j in gradient_updates:
                  gradient_updates[word_j] -= gradient
                  update_count[word_j] += 1
              else:
                  gradient_updates[word_j] = -gradient
                  update_count[word_j] = 1
    
    # VSP term:         
    if hyper_k3 > 0.0:
      for i, (word_i, word_j) in enumerate(vsp_pairs):
          print("\r    Processing VSP pairs {:.1f}%...".format((i+1)/len(vsp_pairs) * 100), flush=True, end='')
  
          original_distance = vsp_pairs[(word_i, word_j)]
          new_distance = distance(new_word_vectors[word_i], new_word_vectors[word_j])
          
          if original_distance <= new_distance: 
  
              gradient = vector_partial_gradient(new_word_vectors[word_i], new_word_vectors[word_j]) 
              gradient = gradient * hyper_k3 
  
              if word_i in gradient_updates:
                  gradient_updates[word_i] -= gradient
                  update_count[word_i] += 1
              else:
                  gradient_updates[word_i] = -gradient
                  update_count[word_i] = 1
  
    print("\r    Applying gradients...", flush=True, end='')
    for word in gradient_updates:
        # we've found that scaling the update term for each word helps with convergence speed. 
        update_term = gradient_updates[word] / (update_count[word]) 
        new_word_vectors[word] += update_term 
    print("\r    Done Applying gradients.", flush=True, end='')
        
    return normalise_word_vectors(new_word_vectors)
  
  
def counter_fit(dct_word_vectors,  synonyms, antonyms, epochs=20, hk1=0.1, hk2=0.1, hk3=0.1, rho=0.2):
  """
  This method repeatedly applies SGD steps to counter-fit word vectors to linguistic constraints. 
  
  https://raw.githubusercontent.com/nmrksic/counter-fitting/master/counterfitting.py
  """
  word_vectors = normalise_word_vectors(dct_word_vectors)
  
  current_iteration = 0
  
  vsp_pairs = {}

  if hk3 > 0.0: # if we need to compute the VSP terms.
      vsp_pairs = compute_vsp_pairs(word_vectors, rho=rho)
  
  # Post-processing: remove synonym pairs which are deemed to be both synonyms and antonyms:
  for antonym_pair in antonyms:
      if antonym_pair in synonyms:
          synonyms.remove(antonym_pair)
      if antonym_pair in vsp_pairs:
          del vsp_pairs[antonym_pair]

  max_iter = epochs
  print("Antonym pairs:", len(antonyms), "Synonym pairs:", len(synonyms), "VSP pairs:", len(vsp_pairs), flush=True)
  print("Running the optimisation procedure for", max_iter, "SGD steps...", flush=True)
  
  while current_iteration < max_iter:
    current_iteration += 1
    print("\r  Counter-fitting SGD step {}...".format(current_iteration), flush=True, end='')
    word_vectors = one_step_SGD(word_vectors, synonyms, antonyms, vsp_pairs, 
                                hyper_k1=hk1, hyper_k2=hk2, hyper_k3=hk3)
  print("")
  return word_vectors  
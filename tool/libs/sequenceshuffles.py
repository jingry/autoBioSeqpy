import numpy as np
from scipy.special import gammaln

def kmers(s, k):
    # iterator over k-mers of sliceable iterable (e.g. string) s
    assert 1 <= k <= len(s)
    for i in range(len(s)-k+1):
        yield s[i:i+k]

def int_tokenize_generator(itr):
    # int tokenize iterable generator
    symbol_to_token = {} # symbol to int token dict
    for x in itr:
        int_token = symbol_to_token.get(x, len(symbol_to_token))
        symbol_to_token[x] = int_token
        yield int_token
        
def int_tokenize(itr, lexicographic_order=False):
    """
    int tokenize iterable
    return dict mapping symbol to int token
    
    Args:
        * itr - iterable
        * lexicographic_order - (bool) iff True sort integer labels lexicographically according to itr
    
    Returns:
        * tokenized - (int list) tokenized
    
    """    
    tokenized = []
    
    symbol_to_token = {} # symbol to int token dict
    for x in itr:
        int_token = symbol_to_token.get(x, len(symbol_to_token))
        symbol_to_token[x] = int_token
        tokenized.append(int_token)
    
    if lexicographic_order:
        K = len(symbol_to_token) # number of distinct symbols
        token_to_symbol = {v:k for (k,v) in symbol_to_token.items()}
        symbol_to_token = dict(zip(sorted(symbol_to_token.keys()), range(K))) # sorted
        token_to_token = {k:symbol_to_token[token_to_symbol[k]] for k in range(K)} # token to sorted token
        n = len(tokenized)
        for i in range(n):
            tokenized[i] = token_to_token[tokenized[i]]
    
    return (tokenized, symbol_to_token)

def transition_counts_from_ints(seq):
    """
    return transition counts matrix N of integer-valued iterable seq
    count N[x,y] = # times have transition from x to y in seq
    assume x in {0,1,...} for all x in s
    
    Args:
        * seq - iterable of ints
        
    Returns:
        * K by K counts (int matrix), where K is max of seq
        
    """
    K = np.max(seq)+1 # number of distinct symbols
    N = np.zeros((K, K), dtype=int)
    for (i,x) in enumerate(seq):
        if i > 0:
            N[x_prev,x] += 1
        x_prev = x
    return N

def counts_from_ints(seq):
    """
    return counts vector C of integer-valued iterable seq
    count C[x] = # times have x in seq
    
    Args:
        * seq - iterable of ints
    
    Returns:
        * K length int vector of counts, where K is max of seq
        
    """
    K = np.max(seq)+1 # number of distinct symbols
    C = np.zeros(K, dtype=int)
    for x in seq:
        C[x] += 1
    return C

def log_size_markov_type_from_transition_counts(N, initial_state, final_state):
    """
    log number of trajectories with Markov type N
    with fixed inital state and final state
    uses Whittle's 1955 formula in notation of Billingsley 1960
    
    Args:
        * N - matrix of transition counts
        * initial_state - int initial state
        * final_state - int final state
        
    Returns:
        * number of trajectories with Markov type N
            starting at initial_state and ending at final_state
    
    """
    
    K = N.shape[0] # alphabet size
    
    # get marginal counts
    N_from = N.sum(axis=1)
    N_to = N.sum(axis=0)
    
    # equation (2.3) in Billingsley 1960
    N_star = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if N_from[i] > 0:
                N_star[i,j] = (i==j) - N[i,j]/N_from[i]
            else:
                N_star[i,j] = (i==j)
    
    N_star_cofactor = np.power(-1, final_state+initial_state) \
        *np.linalg.det(np.delete(np.delete(N_star, initial_state, 1), final_state, 0))

    log_size_type = 0.0
    log_size_type += np.sum(gammaln(N_from[i]+1) for i in range(K))
    log_size_type -= np.sum(gammaln(N[i,j]+1) for i in range(K) for j in range(K))

    log_size_type += np.log(N_star_cofactor)

    return log_size_type

def size_markov_type_from_transition_counts(N, initial_state, final_state):
    # exp of log_size_markov_type    
    return int(np.rint(np.exp(log_size_markov_type_from_transition_counts(N, initial_state, final_state))))

def log_multinomial(mult):
    # log multinomial coefficient (n ; mult[1], mult[2], ...)
    # where n = sum_i mult[i]

    lm = gammaln(sum(mult)+1)
    lm -= sum(gammaln(x+1) for x in mult)
    return lm

def log_num_shuffles_from_string(s, k):
    """
    number of strings having the same counts of substrings of length up to and including k

    Args:
        * s - (string) string
        * k - (int) substring length

    Returns:
        (int) - number of strings

    """
    tokenized = int_tokenize(kmers(s, max(k-1, 1)))[0]
    if k == 1:
        C = counts_from_ints(tokenized)
        return log_multinomial(C)
    else:
        N = transition_counts_from_ints(tokenized)
        return log_size_markov_type_from_transition_counts(N, tokenized[0], tokenized[-1])

def num_shuffles_from_string(s, k):
    """
    log number of strings having the same counts of substrings of length up to and including k

    Args:
        * s - (string) string
        * k - (int) substring length

    Returns:
        (float) - log number of strings

    """
    return int(np.rint(np.exp(log_num_shuffles_from_string(s, k))))

def shuffles_from_transition_counts(transition_counts, initial_state, final_state):
    """
    iterate over sequences having transition count N
    starting in initial_state, ending in final_state
    
    Args:
        * N - int array of transition counts
        * initial_state - int initial state
        * final_state - int final state
    
    Yields:
        * sequence having transition count N, starting in initial_state, ending in final_state
    
    """
    N = transition_counts.copy()
    n = np.sum(N) + 1 # trajectory length
    K = N.shape[0] # number of states
    
    seq = np.zeros(n, dtype=int)-1 # sequence, trajectory
    seq[0] = initial_state
    t = 1
    while t >= 1:        
        prev_state = seq[t-1]
        current_state = seq[t]+1
        while current_state < K and N[prev_state, current_state] == 0:
            current_state += 1
        if current_state == K: # go left
            if seq[t] != -1:
                N[seq[t-1],seq[t]] += 1
            seq[t] = -1
            t -= 1
        else: # go right or stay if at end
            if seq[t] != -1:
                N[seq[t-1],seq[t]] += 1
            seq[t] = current_state
            N[seq[t-1],seq[t]] -= 1
            if t < n-1: # go right if not at end
                t += 1
            else: # stay if at end
                yield tuple(seq)

def distinct_permutations_from_histogram(counts):
    """
    iterate over all distinct permutations of sequences with symbol counts counts
    
    Args:
        * counts - (int vector) histogram of counts
    
    Yields:
        * (int tuple) sequence with histogram counts
    
    """
    C = counts.copy()
    n =  np.sum(C) # trajectory length
    K = len(C) # number of states
    
    seq = np.zeros(n, dtype=int)-1 # sequence, trajectory
    t = 0
    while t >= 0:
        current_state = seq[t]+1
        while current_state < K and C[current_state] == 0:
            current_state += 1
        if current_state == K: # go left
            if seq[t] != -1:
                C[seq[t]] += 1
            seq[t] = -1
            t -= 1
        else: # go right or stay if at end
            if seq[t] != -1:
                C[seq[t]] += 1
            seq[t] = current_state
            C[seq[t]] -= 1
            if t < n-1: # go right if not at end
                t += 1
            else: # stay if at end
                yield tuple(seq)

def shuffles_from_string(s, k, lexicographic_order=False):
    """
    iterate over distinct strings having same counts of substrings of length up to and including k
    
    Args:
        * s - (string) string
        * k - (int) substring length
        * lexicographic_order - (bool) iff True, sort integer labels lexicographically according to itr
        
    Yields:
        * string having same counts of substrings of length up to and including k
    
    Examples:
        >>> list(string_shuffles('catamaran',2))
        ['catamaran', 'cataraman', 'camataran', 'camaratan', 'carataman', 'caramatan']
        
        >>> list(string_shuffles('catamaran',2,lexicographic_order=True))
        ['camaratan', 'camataran', 'caramatan', 'carataman', 'catamaran', 'cataraman']
        
        >>> list(string_shuffles('cat',1,lexicographic_order=False))
        ['cat', 'cta', 'act', 'atc', 'tca', 'tac']
        
        >>> list(string_shuffles('cat',1,lexicographic_order=True))
        ['act', 'atc', 'cat', 'cta', 'tac', 'tca']
        
        >>> list(string_shuffles('caa',1))
        ['caa', 'aca', 'aac']
        
    """
    assert 1 <= k <= len(s)
    
    # tokenize string
    tokenized, symbol_to_token = int_tokenize(\
                                              kmers(s, max(k-1,1)), \
                                              lexicographic_order)
    # int token to symbol
    token_to_symbol = {v:k for (k,v) in symbol_to_token.items()}
    
    if k > 1:
        # int token to last letter of symbol
        token_to_symbol_suffix = {v:k[-1] for (k,v) in symbol_to_token.items()}            
        # get transition counts
        N = transition_counts_from_ints(tokenized)
        
        for seq in shuffles_from_transition_counts(N, tokenized[0], tokenized[-1]):
            # convert int tuple seq to string
            string_seq = token_to_symbol[seq[0]]+''.join(token_to_symbol_suffix[x] for x in seq[1:])
            yield string_seq
            
    elif k == 1: # handle specially
        # get counts
        C = counts_from_ints(tokenized)
        
        for seq in distinct_permutations_from_histogram(C):
            # convert int tuple seq to string
            string_seq = ''.join(token_to_symbol[x] for x in seq)
            yield string_seq


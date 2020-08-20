import pickle
from collections import defaultdict
from scipy import spatial


EMBEDDING_PICKLE = 'embeddings.pkl'

with open(EMBEDDING_PICKLE, 'rb') as fp:
    embedding_dict = pickle.load(fp)


cosine_distances = defaultdict(list)
for id, embeddings in embedding_dict.items():
    for i in range(len(embeddings)):
        e1 = embeddings[i]
        for j in range(i+1, len(embeddings)):
            e2 = embeddings[j]
            cosine_dist = spatial.distance.cosine(e1, e2)
            cosine_distances[id].append(cosine_dist)


print ('Comparing within same identity...')
for id, cd in cosine_distances.items():
    print ('ID {0}'.format(id))
    min_cd = min(cd)
    max_cd = max(cd)
    print ('Min: {0}\tMax: {1}'.format(min_cd, max_cd))


cosine_distances = defaultdict(list)
keys = list(embedding_dict.keys())
num_keys = len(keys)
for index_i in range(num_keys):
    id_i = keys[index_i]
    id_i_embeddings = embedding_dict[id_i]
    for index_j in range(index_i+1, num_keys):
        id_j = keys[index_j]
        id_j_embeddings = embedding_dict[id_j]
        for e1 in id_i_embeddings:
            for e2 in id_j_embeddings:
                cosine_dist = spatial.distance.cosine(e1, e2)
                comparison_key = '{0}-{1}'.format(id_i, id_j)
                cosine_distances[comparison_key].append(cosine_dist)


print ('')
print ('Comparing against different identities...')
for id, cd in cosine_distances.items():
    print ('ID {0}'.format(id))
    min_cd = min(cd)
    max_cd = max(cd)
    print ('Min: {0}\tMax: {1}'.format(min_cd, max_cd))




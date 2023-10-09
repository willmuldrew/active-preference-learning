import torch
from sentence_transformers import SentenceTransformer
# noinspection PyPackageRequirements
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingDiversityRanker:
    def __init__(self):
        self.model = None

    def rank_pairs(self, pairs, old_data):
        if len(old_data) == 0:
            return pairs

        # TODO - should be a cleaner/more efficient way to manage old data so we don't have to compute embeddings repeatedly

        if self.model is None:
            print("Loading sentence transformer for embeddings")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

        old_sentences = []
        for d in old_data:
            # old_sentences.append(d['query_str'] + d['response_str_w'])
            # old_sentences.append(d['query_str'] + d['response_str_l'])
            old_sentences.append(d['response_str_w'])
            old_sentences.append(d['response_str_l'])

        old_embeddings = self.model.encode(old_sentences)

        # TODO - shouldn't have to mention w or l here...  the new pairs are unranked
        # new_sentence_embeddings_a = self.model.encode([p['query_str'] + p['response_str_w'] for p in pairs])
        # new_sentence_embeddings_b = self.model.encode([p['query_str'] + p['response_str_l'] for p in pairs])
        new_sentence_embeddings_a = self.model.encode([p['response_str_w'] for p in pairs])
        new_sentence_embeddings_b = self.model.encode([p['response_str_l'] for p in pairs])

        # NOTE - TODO: this doesn't select for diversity between samples in our new batch... could this result in them being not that diverse within the batch?
        similarities = torch.tensor(
            (cosine_similarity(new_sentence_embeddings_a, old_embeddings).mean(axis=1) + cosine_similarity(new_sentence_embeddings_b, old_embeddings).mean(axis=1)) / 2.0)

        res = [pairs[i] for i in torch.argsort(-similarities)]
        return res



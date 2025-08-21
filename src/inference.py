from sentence_transformers import SentenceTransformer, util

def inference_from_hub(model_id, instruction, query, passages):
    hub_model = SentenceTransformer(model_id)
    instructed_query = instruction + query
    query_embedding = hub_model.encode(instructed_query)
    passage_embeddings = hub_model.encode(passages)
    similarities = util.cos_sim(query_embedding, passage_embeddings)
    return [(float(similarities[0][i]), passages[i]) for i in range(len(passages))]

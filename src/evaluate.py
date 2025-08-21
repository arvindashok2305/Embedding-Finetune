from sentence_transformers import util

def evaluate_model(model, instruction, query, passages):
    instructed_query = instruction + query
    query_embedding = model.encode(instructed_query)
    passage_embeddings = model.encode(passages)
    similarities = util.cos_sim(query_embedding, passage_embeddings)

    results = [(float(similarities[0][i]), passages[i]) for i in range(len(passages))]
    return results

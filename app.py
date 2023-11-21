import numpy as np
from gensim.models import KeyedVectors
from flask import Flask, request, jsonify

app = Flask(__name__)
app.config["DEBUG"] = True

class Model:
    def __init__(self):
        self.uni_model = KeyedVectors.load("./models/unigram.model")
        self.bi_model = KeyedVectors.load("./models/bigram.model")
        self.tri_model = KeyedVectors.load("./models/trigram.model")

model = Model()

def infer(target):
    sim_word_scores = get_sim_skills(target, model.tri_model)
    if sim_word_scores == -1:
        sim_word_scores = get_sim_skills(target, model.bi_model)
        if sim_word_scores == -1:
            target = target.replace("_", " ") if "_" in target else target
            sim_word_scores = get_sim_skills(target, model.uni_model)
            if sim_word_scores == -1:
                return "keyword NOT in the corpus!"
    return [x.replace("_", " ") for x in sim_word_scores]

def concat_words(target):
    # target = "system administrator"
    target = target.lower()
    if len(target) > 1:
        target = target.split()
        target = "_".join(target)
    return target

def get_sim_skills(phrase, w2v_model):
    # Tokenize the phrase
    words = phrase.split()
    word_vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    
    if word_vectors:
        # Compute the mean vector of all word vectors
        phrase_vector = np.mean(word_vectors, axis=0)
        sim_word_scores = w2v_model.wv.most_similar(positive=[phrase_vector], topn=50)
        sim_words = []
        for i in range(len(words), len(sim_word_scores)):
            sim_words.append(sim_word_scores[i][0])
        return sim_words    
    else:
        print(f"No words in {w2v_model}!")
        return -1    

@app.route('/skills-explorer', methods=['POST'])
def skills_explorer():
    try:
        skill = request.get_json()
        skill = concat_words(skill['keyword'])
        sim_words = infer(skill)
        return jsonify({'Similar keywords': sim_words})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(port=6666)
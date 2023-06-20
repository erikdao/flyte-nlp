import os
import random
import typing
from dataclasses import dataclass
from typing import Dict, List

import flytekit
import gensim
import nltk
import numpy as np
import plotly.graph_objects as go
import plotly.io as io
from dataclasses_json import dataclass_json
from flytekit import Resources, task, workflow
from flytekit.types.file import FlyteFile
from loguru import logger
from gensim import utils
from gensim.corpora import Dictionary
from gensim.models import LdaModel, Word2Vec
from gensim.parsing.preprocessing import STOPWORDS, remove_stopwords
from gensim.test.utils import datapath
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.manifold import TSNE


MODELSER_NLP = typing.TypeVar("model")
model_file = typing.NamedTuple("ModelFile", model=FlyteFile[MODELSER_NLP])

data_dir = os.path.join(gensim.__path__[0], "test", "test_data")
lee_train_file = os.path.join(data_dir, "lee_background.cor")

# Define types that will be used for Flyte tasks' output
plotdata = typing.NamedTuple(
    "PlottingData",
    x_values=List[float],
    y_values=List[float],
    labels=List[str],
)

workflow_outputs = typing.NamedTuple(
    "WorkflowOutputs",
    simwords=Dict[str, float],
    distance=float,
    topics=Dict[int, List[str]],
)

SENTENCE_A = "Australian cricket captain has supported fast bowler"
SENTENCE_B = "Fast bowler received support from cricket captain"

##############################################################################
# Data Generation
##############################################################################

def pre_processing(line: str) -> List[str]:
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(remove_stopwords(line.lower()))
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __init__(self, path):
        self.corpus_path = datapath(path)

    def __iter__(self):
        for line in open(self.corpus_path):
            yield pre_processing(line)


@task
def generate_processed_corpus() -> List[List[str]]:
    # download the required packages from the nltk library
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    sentences_train = MyCorpus(lee_train_file)
    train_corpus = list(sentences_train)
    return train_corpus


##############################################################################
# Hyperparameters
##############################################################################
@dataclass_json
@dataclass
class Word2VecModelHyperparams(object):
    """
    Hyperparameters that can be used while training the word2vec model.
    """

    vector_size: int = 200
    min_count: int = 1
    workers: int = 4
    compute_loss: bool = True


@dataclass_json
@dataclass
class LDAModelHyperparams(object):
    """
    Hyperparameters that can be used while training the LDA model.
    """

    num_topics: int = 5
    alpha: str = "auto"
    passes: int = 10
    chunksize: int = 100
    update_every: int = 1
    random_state: int = 100


##############################################################################
# Training
##############################################################################
@task
def train_word2vec_model(
    training_data: List[List[str]], hyperparams: Word2VecModelHyperparams
) -> model_file:

    model = Word2Vec(
        training_data,
        min_count=hyperparams.min_count,
        workers=hyperparams.workers,
        vector_size=hyperparams.vector_size,
        compute_loss=hyperparams.compute_loss,
    )
    training_loss = model.get_latest_training_loss()
    logger.info(f"training loss: {training_loss}")
    out_path = os.path.join(
        flytekit.current_context().working_directory, "word2vec.model"
    )
    model.save(out_path)
    return (out_path,)


@task
def train_lda_model(
    corpus: List[List[str]], hyperparams: LDAModelHyperparams
) -> Dict[int, List[str]]:
    id2word = Dictionary(corpus)
    bow_corpus = [id2word.doc2bow(doc) for doc in corpus]
    id_words = [[(id2word[id], count) for id, count in line] for line in bow_corpus]
    logger.info(f"Sample of bag of words generated: {id_words[:2]}")
    lda = LdaModel(
        corpus=bow_corpus,
        id2word=id2word,
        num_topics=hyperparams.num_topics,
        alpha=hyperparams.alpha,
        passes=hyperparams.passes,
        chunksize=hyperparams.chunksize,
        update_every=hyperparams.update_every,
        random_state=hyperparams.random_state,
    )
    return dict(lda.show_topics(num_words=5))


##############################################################################
# Word Similarities
##############################################################################
@task(cache_version="1.0", cache=True, limits=Resources(mem="600Mi"))
def word_similarities(
    model_ser: FlyteFile[MODELSER_NLP], word: str
) -> Dict[str, float]:
    model = Word2Vec.load(model_ser.download())
    wv = model.wv
    logger.info(f"Word vector for {word}:{wv[word]}")
    return dict(wv.most_similar(word, topn=10))


##############################################################################
# Sentence Similarities
##############################################################################
@task(cache_version="1.0", cache=True, limits=Resources(mem="600Mi"))
def word_movers_distance(model_ser: FlyteFile[MODELSER_NLP]) -> float:
    sentences = [SENTENCE_A, SENTENCE_B]
    results = []
    for i in sentences:
        result = [w for w in utils.tokenize(i) if w not in STOPWORDS]
        results.append(result)
    model = Word2Vec.load(model_ser.download())
    logger.info(f"Computing word movers distance for: {SENTENCE_A} and {SENTENCE_B} ")
    return model.wv.wmdistance(*results)


@task(cache_version="1.0", cache=True, limits=Resources(mem="1000Mi"))
def dimensionality_reduction(model_ser: FlyteFile[MODELSER_NLP]) -> plotdata:
    model = Word2Vec.load(model_ser.download())
    num_dimensions = 2
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)
    logger.info("Running dimensionality reduction using t-SNE")
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)
    x_vals = [float(v[0]) for v in vectors]
    y_vals = [float(v[1]) for v in vectors]
    labels = [str(l) for l in labels]
    return x_vals, y_vals, labels


@task(
    cache_version="1.0", cache=True, limits=Resources(mem="600Mi"), disable_deck=False
)
def plot_with_plotly(x: List[float], y: List[float], labels: List[str]):
    layout = go.Layout(height=600, width=800)
    fig = go.Figure(
        data=go.Scattergl(x=x, y=y, mode="markers", marker=dict(color="aqua")),
        layout=layout,
    )
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 50)
    for i in selected_indices:
        fig.add_annotation(
            text=labels[i],
            x=x[i],
            y=y[i],
            showarrow=False,
            font=dict(size=15, color="black", family="Sans Serif"),
        )
    logger.info("Generating the Word Embedding Plot using Flyte Deck")
    flytekit.Deck("Word Embeddings", io.to_html(fig, full_html=True))


@workflow
def nlp_workflow(target_word: str = "computer") -> workflow_outputs:
    corpus = generate_processed_corpus()
    model_wv = train_word2vec_model(
        training_data=corpus, hyperparams=Word2VecModelHyperparams()
    )
    lda_topics = train_lda_model(corpus=corpus, hyperparams=LDAModelHyperparams())
    similar_words = word_similarities(model_ser=model_wv.model, word=target_word)
    distance = word_movers_distance(model_ser=model_wv.model)
    axis_labels = dimensionality_reduction(model_ser=model_wv.model)
    plot_with_plotly(
        x=axis_labels.x_values, y=axis_labels.y_values, labels=axis_labels.labels
    )
    return similar_words, distance, lda_topics


if __name__ == "__main__":
    logger.info(f"Running {__file__} main...")
    logger.info(nlp_workflow())

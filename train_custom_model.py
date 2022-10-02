from classifier import load_data,tokenize, compute_word_idf
from classifier import custom_feature_extractor, classifier_agent


import numpy as np


def main():
    print("Creating a classifier agent:")

    with open('data/vocab.txt') as file:
        reading = file.readlines()
        vocab_list = [item.strip() for item in reading]
        vocab_dict = {item: i for i, item in enumerate(vocab_list)}

    print("Loading and processing data ...")

    sentences_pos = load_data("data/training_pos.txt")
    sentences_neg = load_data("data/training_neg.txt")

    train_sentences = sentences_pos + sentences_neg

    train_labels = [1 for i in range(len(sentences_pos))] + [0 for i in range(len(sentences_neg))]

    sentences_pos = load_data("data/test_pos_public.txt")
    sentences_neg = load_data("data/test_neg_public.txt")
    test_sentences = sentences_pos + sentences_neg
    test_labels = [1 for i in range(len(sentences_pos))] + [0 for i in range(len(sentences_neg))]

    word_idf = compute_word_idf(train_sentences,vocab_list)

    # TODO: ====================== Your code here ====================

    feat_map = custom_feature_extractor(vocab_list, tokenize)
    # You many replace this with a different feature extractor

    # feat_map = tfidf_extractor(vocab_list, tokenize, word_freq)

    # TODO: ==========================================================

    # train with SGD
    nepoch = 10
    print("Training using SGD for ", nepoch, "data passes.")
    d = len(vocab_list)
    params = np.array([0.0 for i in range(d)])
    custom_classifier = classifier_agent(feat_map, params)

    # TODO: ====================== Feel free to tweak how it is trained here====================
    custom_classifier.train_sgd(train_sentences, train_labels, nepoch, 0.001)

    # niter = 1000
    # custom_classifier.train_gd(train_sentences, train_labels, niter, 0.01)

    err = custom_classifier.eval_model(test_sentences,test_labels)

    print("Test error =  ", err)

    custom_classifier.save_params_to_file('best_model.npy')


if __name__ == "__main__":
    main()
from classifier import load_data,tokenize, feature_extractor, classifier_agent

import numpy as np
import matplotlib.pyplot as plt


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


    feat_map = feature_extractor(vocab_list, tokenize)
    # You many replace this with a different feature extractor

    # feat_map = tfidf_extractor(vocab_list, tokenize, word_freq)

    # train with GD
    niter = 100
    print("Training using GD for ", niter, "iterations.")
    d = len(vocab_list)
    #print(len(vocab_list))
    params = np.array([0.0 for i in range(d)])
    #print(len(params))
    classifier1 = classifier_agent(feat_map,params)
    losses, errors = classifier1.train_gd(train_sentences,train_labels,niter,0.01)
    #print(len(classifier1.params))
    print('Losses:',losses)
    print('Errors:',errors)

    #classifier1.save_params_to_file('best_model.npy')

    #print(len(np.load('best_model.npy')))

    #plt.plot(list(range(len(errors))),errors, label="GD Error")
    #my_sentence = "This movie is amazing! Truly a masterpiece."
    #ypred = classifier1.predict(my_sentence,RAW_TEXT=True)
    #print(ypred)


    

    
    # train with SGD
    nepoch = 10
    print("\nTraining using SGD for ", nepoch, "data passes.")
    d = len(vocab_list)
    params = np.array([0.0 for i in range(d)])
    classifier2 = classifier_agent(feat_map, params)
    losses, errors = classifier2.train_sgd(train_sentences, train_labels, nepoch, 0.001)
    print('Losses:',losses)
    print('Errors:',errors)

    #plt.plot(list(range(len(errors))),errors, label="SGD Error")
    #plt.legend()
    #plt.show()
    
    err1 = classifier1.eval_model(test_sentences,test_labels)
    err2 = classifier2.eval_model(test_sentences,test_labels)

    print('GD: test err = ', err1,
          'SGD: test err = ', err2)
    classifier1.save_params_to_file('best_model.npy')

if __name__ == "__main__":
    main()
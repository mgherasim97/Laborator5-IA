from sklearn import preprocessing
import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score, classification_report

#exercise 2
def normalize_data (train_data, test_data, type = None):

    if   type == "standard":
        scaler = preprocessing.StandardScaler()
    elif type == "min_max":
        scaler = preprocessing.minmax_scale()
    elif type == "l1":
        scaler = preprocessing.Normalizer(norm = "l1")
    elif type == "l2":
        scaler = preprocessing.Normalizer(norm = "l2")


    scaler.fit(train_data)
    scaled_training = scaler.transform(train_data)
    scaled_test     = scaler.transform(test_data)
    return (scaled_training, scaled_test)

#exercise 3
class bow:
    def __init__ (self):
        self.vocab = { }
        self.words = [ ]
        self.len   = 0

    #exercise 3'
    def build_vocabulary (self, data):
        for doc in data:
            for word in doc:
                if word not in self.words:
                    self.vocab[word] = self.len
                    self.words.append(word)
                    self.len        += 1
        return self.len

    #exercise 4
    def get_features(self, data):
        features = np.zeros((data.shape[0], self.len))
        for sample_idx, sentence in enumerate(data):
            for word in sentence:
                if word in self.vocab:
                    features[sample_idx, self.vocab[word]] += 1
        return features

def best_worst_spam(classifier, bow):
    #we flatten the array and sort them by incides
    spam_values = np.ravel(classifier.coef_)
    indices = np.argsort(spam_values)

    #WE CONVERT the words to an integer scalar np.array
    integer_words = np.array(bow.words)
    print('NON-SPAM PLS STAY', integer_words[indices[:10]])
    print('SPAM  PLS GO AWAY', integer_words[indices[-10:]])



train_data   = np.load("../data/training_sentences.npy",
                       allow_pickle = True)
train_labels = np.load("../data/training_labels.npy",
                       allow_pickle = True)
test_data    = np.load("../data/test_sentences.npy",
                       allow_pickle = True)
test_labels  = np.load("../data/test_labels.npy",
                       allow_pickle = True)

#initialize and train the model
bag_of_words_model = bow()
print(bag_of_words_model.build_vocabulary(train_data))



#convert the text variables to numbers
training_features = bag_of_words_model.get_features(train_data)
testing_features  = bag_of_words_model.get_features(test_data )



#exercise 5
(norm_train, norm_test) = normalize_data(training_features,
                                         testing_features,
                                         type = "l2")

#exercise 6
classifier       = svm.SVC(C = 1.0, kernel = "linear")
classifier.fit(norm_train, train_labels)
test_predictions = classifier.predict(norm_test)
acc_score = accuracy_score(test_labels, test_predictions)
f_score = f1_score(test_labels, test_predictions)

print("Accuracy score:{}".format(acc_score))
print("F1 score:{}"      .format(f_score))

best_worst_spam(classifier, bag_of_words_model)

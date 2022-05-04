from datasets import load_dataset
import nltk

if __name__ == '__main__':
    load_dataset("wikipedia", "20220301.simple")
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

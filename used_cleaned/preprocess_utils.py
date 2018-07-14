from toxic_comments.used_cleaned.global_variables import Text

class Preprocessor():

    def lower(text):
        text = text.lower()
        return text

def preprocess(data):

    print('preprocessing')
    p = Preprocessor()
    print('lowercase')
    data[Text] = data[Text].map(lambda x: p.lower(x))

    return data
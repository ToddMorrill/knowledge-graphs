from allennlp.predictors.predictor import Predictor


class AllenNLPCoref(object):
    def __init__(self) -> None:
        self.predictor = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
        )

    def predict(self, document):
        return self.predictor.predict(document)

    def coref_resolved(self, document):
        return self.predictor.coref_resolved(document)

    def custom_coref_resolved(self, document):
        # use a named entity as the anchor entity
        # otherwise just use first entity
        raise NotImplementedError


def main():
    text = (
        "Paul Allen was born on January 21, 1953, in Seattle, Washington, to Kenneth Sam Allen and Edna Faye Allen. "
        "Allen attended Lakeside School, a private school in Seattle, where he befriended Bill Gates, two years younger, "
        "with whom he shared an enthusiasm for computers. Paul and Bill used a teletype terminal at their high school, "
        "Lakeside, to develop their programming skills on several time-sharing computer systems."
    )

    predictor = AllenNLPCoref()
    result = predictor.predict(document=text)

    print(predictor.coref_resolved(text))


if __name__ == '__main__':
    main()
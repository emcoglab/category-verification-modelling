from framework.cognitive_model.ldm.corpus.tokenising import modified_word_tokenize
from framework.data.category_verification_data import CategoryVerificationItemData, Filter


def main():
    df = CategoryVerificationItemData().dataframe_filtered(
        with_filter=Filter("differently assumed", assumed_object_label_differs=True, repeated_items_tokeniser=modified_word_tokenize))

    df.to_csv("~/Desktop/test.csv")


if __name__ == '__main__':
    main()

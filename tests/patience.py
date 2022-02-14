

def test_dataset():
    from datasets import Dataset
    patience = Dataset("config/patience.json")
    pp10 = patience("pp10")
    print(len(pp10))
    print(pp10.input_dim, pp10.output_dim)


def test_patience():
    from datasets import Patience
    patience = Patience()
    pp10 = patience("pp10")
    print(len(pp10))
    print(pp10.input_dim, pp10.output_dim)


if __name__ == "__main__":
    test_dataset()
    test_patience()